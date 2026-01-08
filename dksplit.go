package dksplit

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	padIdx  = 0
	unkIdx  = 1
	maxLen  = 64
	numTags = 2
)

var charMap [128]int64

// Singleton for ORT initialization
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

func initORT() error {
	ortOnce.Do(func() {
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
	err := initORT()
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

// SplitBatch segments multiple strings with length grouping for efficiency
func (s *Splitter) SplitBatch(texts []string, batchSize int) ([][]string, error) {
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

			emissions, err := s.runInference(charIds, batchLen, length)
			if err != nil {
				return nil, err
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