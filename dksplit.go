package dksplit

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

// errInvalidK is returned by SplitTopK when k < 1.
var errInvalidK = errors.New("dksplit: k must be >= 1")

const (
	padIdx  = 0
	unkIdx  = 1
	maxLen  = 64
	numTags = 2
)

var charMap [128]int64

var (
	ortOnce sync.Once
	ortErr  error
)

func init() {
	for i := range charMap {
		charMap[i] = unkIdx
	}
	vocab := "abcdefghijklmnopqrstuvwxyz0123456789"
	for i, c := range vocab {
		charMap[c] = int64(i + 2)
	}
}

func initORT(modelDir string) error {
	ortOnce.Do(func() {
		libPath := filepath.Join(modelDir, "libonnxruntime.so")
		ort.SetSharedLibraryPath(libPath)
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// Splitter is the main word segmentation engine
type Splitter struct {
	session          *ort.DynamicAdvancedSession
	transitions      []float32
	startTransitions []float32
	endTransitions   []float32
}

// New creates a new Splitter instance
func New(modelDir string) (*Splitter, error) {
	err := initORT(modelDir)
	if err != nil {
		return nil, err
	}

	modelPath := filepath.Join(modelDir, "dksplit-int8.onnx")

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"chars"},
		[]string{"emissions"},
		nil,
	)
	if err != nil {
		return nil, err
	}

	transitions, err := loadFloat32Bin(filepath.Join(modelDir, "transitions.bin"))
	if err != nil {
		return nil, err
	}

	startTrans, err := loadFloat32Bin(filepath.Join(modelDir, "start_transitions.bin"))
	if err != nil {
		return nil, err
	}

	endTrans, err := loadFloat32Bin(filepath.Join(modelDir, "end_transitions.bin"))
	if err != nil {
		return nil, err
	}

	return &Splitter{
		session:          session,
		transitions:      transitions,
		startTransitions: startTrans,
		endTransitions:   endTrans,
	}, nil
}

// Split segments a single string into words
func (s *Splitter) Split(text string) ([]string, error) {
	if len(text) == 0 {
		return []string{}, nil
	}

	text = strings.ToLower(text)
	if len(text) > maxLen {
		text = text[:maxLen]
	}

	seqLen := len(text)
	charIds := textToIds(text)

	emissions, err := s.runInference(charIds, 1, seqLen)
	if err != nil {
		return nil, err
	}

	preds := s.crfDecodeBatch(emissions, 1, seqLen)

	return decodeToWords(text, preds[0]), nil
}

// SplitTopK segments a single string and returns the top-k segmentations,
// best first. Distinct CRF tag paths can map to the same segmentation, so a
// beam of 2k paths is decoded and deduplicated. Short inputs may yield fewer
// than k candidates. The rank-1 candidate always equals Split.
func (s *Splitter) SplitTopK(text string, k int) ([][]string, error) {
	if k < 1 {
		return nil, errInvalidK
	}
	if len(text) == 0 {
		return [][]string{}, nil
	}

	text = strings.ToLower(text)
	if len(text) > maxLen {
		text = text[:maxLen]
	}

	seqLen := len(text)
	charIds := textToIds(text)

	emissions, err := s.runInference(charIds, 1, seqLen)
	if err != nil {
		return nil, err
	}

	// Each segmentation corresponds to exactly 2 tag paths (the first
	// character's tag does not affect word boundaries), so a beam of 2k
	// paths is always enough to yield k unique segmentations.
	paths := s.crfDecodeTopK(emissions, seqLen, 2*k)

	results := make([][]string, 0, k)
	seen := make(map[string]struct{}, 2*k)
	for _, path := range paths {
		words := decodeToWords(text, path)
		key := strings.Join(words, "\x00")
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		results = append(results, words)
		if len(results) == k {
			break
		}
	}

	return results, nil
}

// Split3 returns the top-3 segmentations, best first.
func (s *Splitter) Split3(text string) ([][]string, error) {
	return s.SplitTopK(text, 3)
}

// Split5 returns the top-5 segmentations, best first.
func (s *Splitter) Split5(text string) ([][]string, error) {
	return s.SplitTopK(text, 5)
}

// SplitBatch segments multiple strings, grouping inputs by length for
// efficiency. Inference runs row by row, so each result is guaranteed
// identical to Split on that text. For maximum throughput at the cost of
// that guarantee, see SplitBatchFast.
func (s *Splitter) SplitBatch(texts []string, batchSize int) ([][]string, error) {
	return s.splitBatch(texts, batchSize, true)
}

// SplitBatchFast segments multiple strings using whole-batch inference,
// roughly 2-4x faster than SplitBatch. The INT8 model is dynamically
// quantized, so activation scales are computed over the whole batch tensor,
// and the result for a string can differ slightly from Split depending on
// the other strings batched with it.
func (s *Splitter) SplitBatchFast(texts []string, batchSize int) ([][]string, error) {
	return s.splitBatch(texts, batchSize, false)
}

func (s *Splitter) splitBatch(texts []string, batchSize int, exact bool) ([][]string, error) {
	if len(texts) == 0 {
		return [][]string{}, nil
	}

	if batchSize <= 0 {
		batchSize = 256
	}

	n := len(texts)
	results := make([][]string, n)

	type item struct {
		index int
		text  string
	}

	lengthGroups := make(map[int][]item)

	for i, text := range texts {
		processed := strings.ToLower(text)
		if len(processed) > maxLen {
			processed = processed[:maxLen]
		}

		length := len(processed)
		if length == 0 {
			results[i] = []string{}
		} else {
			lengthGroups[length] = append(lengthGroups[length], item{i, processed})
		}
	}

	lengths := make([]int, 0, len(lengthGroups))
	for l := range lengthGroups {
		lengths = append(lengths, l)
	}
	sort.Ints(lengths)

	for _, length := range lengths {
		group := lengthGroups[length]

		for batchStart := 0; batchStart < len(group); batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > len(group) {
				batchEnd = len(group)
			}

			batch := group[batchStart:batchEnd]
			batchLen := len(batch)

			charIds := make([]int64, batchLen*length)
			batchTexts := make([]string, batchLen)

			for i, it := range batch {
				batchTexts[i] = it.text
				ids := textToIds(it.text)
				copy(charIds[i*length:], ids)
			}

			var emissions []float32
			if exact {
				// Row by row keeps results identical to Split: the INT8
				// model is dynamically quantized, so in a whole-batch run
				// rows perturb each other's emissions.
				emissions = make([]float32, 0, batchLen*length*numTags)
				for i := 0; i < batchLen; i++ {
					rowEmissions, err := s.runInference(charIds[i*length:(i+1)*length], 1, length)
					if err != nil {
						return nil, err
					}
					emissions = append(emissions, rowEmissions...)
				}
			} else {
				batchEmissions, err := s.runInference(charIds, batchLen, length)
				if err != nil {
					return nil, err
				}
				emissions = batchEmissions
			}

			preds := s.crfDecodeBatch(emissions, batchLen, length)

			for i, it := range batch {
				results[it.index] = decodeToWords(batchTexts[i], preds[i])
			}
		}
	}

	return results, nil
}

// Close releases resources
func (s *Splitter) Close() error {
	if s.session != nil {
		return s.session.Destroy()
	}
	return nil
}

func textToIds(text string) []int64 {
	ids := make([]int64, len(text))
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c < 128 {
			ids[i] = charMap[c]
		} else {
			ids[i] = unkIdx
		}
	}
	return ids
}

func (s *Splitter) runInference(charIds []int64, batchSize, seqLen int) ([]float32, error) {
	inputShape := ort.Shape{int64(batchSize), int64(seqLen)}
	inputTensor, err := ort.NewTensor(inputShape, charIds)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputShape := ort.Shape{int64(batchSize), int64(seqLen), numTags}
	outputData := make([]float32, batchSize*seqLen*numTags)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	err = s.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, err
	}

	return outputData, nil
}

func (s *Splitter) crfDecodeBatch(emissions []float32, batchSize, seqLen int) [][]int {
	results := make([][]int, batchSize)

	for b := 0; b < batchSize; b++ {
		offset := b * seqLen * numTags

		score := make([]float32, numTags)
		for i := 0; i < numTags; i++ {
			score[i] = s.startTransitions[i] + emissions[offset+i]
		}

		history := make([][]int, seqLen-1)

		for t := 1; t < seqLen; t++ {
			history[t-1] = make([]int, numTags)
			newScore := make([]float32, numTags)
			emitOffset := offset + t*numTags

			for j := 0; j < numTags; j++ {
				maxScore := float32(-math.MaxFloat32)
				maxIdx := 0
				for i := 0; i < numTags; i++ {
					sc := score[i] + s.transitions[i*numTags+j] + emissions[emitOffset+j]
					if sc > maxScore {
						maxScore = sc
						maxIdx = i
					}
				}
				newScore[j] = maxScore
				history[t-1][j] = maxIdx
			}
			score = newScore
		}

		bestLast := 0
		bestScore := float32(-math.MaxFloat32)
		for i := 0; i < numTags; i++ {
			sc := score[i] + s.endTransitions[i]
			if sc > bestScore {
				bestScore = sc
				bestLast = i
			}
		}

		path := make([]int, seqLen)
		path[seqLen-1] = bestLast
		for t := seqLen - 2; t >= 0; t-- {
			path[t] = history[t][path[t+1]]
		}

		results[b] = path
	}

	return results
}

// crfDecodeTopK performs k-best Viterbi decoding for a single sequence.
// It keeps the k best-scoring paths per tag at each step, then returns all
// finite-score paths sorted by total score, best first. At most k*numTags
// paths are returned. emissions is laid out as (seqLen, numTags) for one
// sequence (batch index 0).
func (s *Splitter) crfDecodeTopK(emissions []float32, seqLen, k int) [][]int {
	const negInf = float32(-math.MaxFloat32)

	// score[j*k+r] = score of the r-th best path ending at tag j.
	// newScore is a separate buffer so that within one time step every
	// target tag reads the previous step's scores; updating score in
	// place would let later tags see this step's freshly written values.
	score := make([]float32, numTags*k)
	newScore := make([]float32, numTags*k)
	for i := range score {
		score[i] = negInf
	}
	for j := 0; j < numTags; j++ {
		score[j*k] = s.startTransitions[j] + emissions[j]
	}

	// history[t-1][j*k+r] = flat index (prevTag*k + prevRank) of the
	// predecessor of the r-th best path ending at tag j at step t.
	history := make([][]int, seqLen-1)

	// Reusable candidate buffer: for each target tag j, the numTags*k
	// candidate scores coming from every (prevTag, prevRank).
	cand := make([]float32, numTags*k)
	candIdx := make([]int, numTags*k)

	for t := 1; t < seqLen; t++ {
		hist := make([]int, numTags*k)
		emitOffset := t * numTags

		for j := 0; j < numTags; j++ {
			// Build all numTags*k candidates for target tag j.
			for i := 0; i < numTags; i++ {
				trans := s.transitions[i*numTags+j]
				emit := emissions[emitOffset+j]
				for r := 0; r < k; r++ {
					prev := score[i*k+r]
					idx := i*k + r
					if prev == negInf {
						cand[idx] = negInf
					} else {
						// Accumulate in the same order as crfDecodeBatch,
						// (prev + trans) + emit, so float32 rounding agrees
						// and the rank-1 path always equals Split.
						cand[idx] = prev + trans + emit
					}
					candIdx[idx] = idx
				}
			}

			// Partial top-k selection of candidates by score, descending.
			selectTopK(cand, candIdx, k)

			for r := 0; r < k; r++ {
				newScore[j*k+r] = cand[candIdx[r]]
				hist[j*k+r] = candIdx[r]
			}
		}
		score, newScore = newScore, score
		history[t-1] = hist
	}

	// Final scores: add end transitions, then sort all numTags*k entries.
	type scored struct {
		flat  int
		value float32
	}
	finals := make([]scored, 0, numTags*k)
	for j := 0; j < numTags; j++ {
		for r := 0; r < k; r++ {
			v := score[j*k+r]
			if v == negInf {
				continue
			}
			finals = append(finals, scored{flat: j*k + r, value: v + s.endTransitions[j]})
		}
	}
	sort.SliceStable(finals, func(a, b int) bool {
		return finals[a].value > finals[b].value
	})

	paths := make([][]int, 0, len(finals))
	for _, f := range finals {
		j := f.flat / k
		r := f.flat % k
		path := make([]int, seqLen)
		path[seqLen-1] = j
		for t := seqLen - 1; t > 0; t-- {
			flat := history[t-1][j*k+r]
			j = flat / k
			r = flat % k
			path[t-1] = j
		}
		paths = append(paths, path)
	}

	return paths
}

// selectTopK reorders the first k entries of idx so that values[idx[0..k]]
// are the k largest of values[idx[...]], in descending order. Uses a simple
// partial selection sort; k and len(idx) are tiny (numTags*k, k <= ~10).
func selectTopK(values []float32, idx []int, k int) {
	n := len(idx)
	if k > n {
		k = n
	}
	for a := 0; a < k; a++ {
		best := a
		for b := a + 1; b < n; b++ {
			if values[idx[b]] > values[idx[best]] {
				best = b
			}
		}
		idx[a], idx[best] = idx[best], idx[a]
	}
}

func decodeToWords(text string, preds []int) []string {
	var words []string
	var current strings.Builder

	for i, c := range text {
		if preds[i] == 1 && current.Len() > 0 {
			words = append(words, current.String())
			current.Reset()
		}
		current.WriteRune(c)
	}

	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

func loadFloat32Bin(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	n := len(data) / 4
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		result[i] = float32frombits(bits)
	}
	return result, nil
}

func float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}