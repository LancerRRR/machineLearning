package nn

import (
	"fmt"
	"machineLearning/util"
	"math"
)

type Layer interface {
	Init()
	FeedFoward(input matrix.Matrix) matrix.Matrix
	BackWard(gradientNext matrix.Matrix) matrix.Matrix
	Update(lr float64)
}

type Linear struct {
	Weights        matrix.Matrix
	Inputs         matrix.Matrix
	Bias           matrix.Matrix
	GradientWeight matrix.Matrix
	GradientBias   matrix.Matrix
	InSize         int
	OutSize        int
}

type Sigmoid struct {
	OutPut matrix.Matrix
}

type BinaryCrossEntropy struct {
	InputY matrix.Matrix
	InputT matrix.Matrix
}

type Net struct {
	Sequential []Layer
}

// Init weights and bias
func (l *Linear) Init() {
	weights := make([][]float64, 0)
	for i := 0; i < l.InSize; i++ {
		weight := []float64{}
		for j := 0; j < l.OutSize; j++ {
			weight = append(weight, 1)
		}
		weights = append(weights, weight)
	}
	bias := [][]float64{}
	for i := 0; i < l.OutSize; i++ {
		bias = append(bias, []float64{0.5})
	}
	//a := matrix.InitMatrix(weights)
	fmt.Println(weights)
	l.Weights = matrix.InitMatrix(weights)
	l.Bias = matrix.InitMatrix(bias)
}

// y = Xw + b
func (l *Linear) FeedFoward(input matrix.Matrix) (out matrix.Matrix) {
	out = input.Multiply(l.Weights).ApplyWithVector(l.Bias.Transpose(), func(a, b float64) float64 { return a + b })
	l.Inputs = input
	return
}

// Backpropagation of fully-connected layer
func (l *Linear) BackWard(gradiantNext matrix.Matrix) matrix.Matrix {
	l.GradientWeight = l.Inputs.Transpose().Multiply(gradiantNext)
	l.GradientBias = gradiantNext.SumAlongAxis(0).Transpose()
	return gradiantNext.Multiply(l.Weights.Transpose())
}

// update weights and bias
func (l *Linear) Update(lr float64) {
	l.Weights = l.Weights.Apply2Matrix(l.GradientWeight, func(a, b float64) float64 {
		return a - lr*b
	})
	l.Bias = l.Bias.Apply2Matrix(l.Bias, func(a, b float64) float64 {
		return a - lr*b
	})
}

func (s *Sigmoid) Init() {
}

func (s *Sigmoid) FeedFoward(input matrix.Matrix) (out matrix.Matrix) {
	out = input.Apply(sigmoid)
	s.OutPut = out
	return
}

// Backpropagation of sigmoid layer
func (s *Sigmoid) BackWard(gradiantNext matrix.Matrix) matrix.Matrix {
	part1 := gradiantNext.Apply2Matrix(s.OutPut, func(a, b float64) float64 {
		return a * b
	})
	// (1-output)
	part2 := s.OutPut.Apply(func(a float64) float64 {
		return 1 - a
	})
	return part1.Apply2Matrix(part2, func(a, b float64) float64 {
		return a * b
	})
}

func (s *Sigmoid) Update(lr float64) {
	// Nothing to update in sigmoid layer
}

func sigmoid(a float64) float64 {
	return 1 / (1 + math.Exp(-a))
}

func (b *BinaryCrossEntropy) Loss(y, t matrix.Matrix) float64 {
	b.InputY = y
	b.InputT = t
	n := y.RowSize
	//  loss = -(t*log(y) + (1-t)*log(1-y))
	// part1: t * log(y)
	part1 := t.Transpose().Multiply(y.Apply(func(a float64) float64 {
		return math.Log(a)
	}))
	// part2 : 1-t
	part2 := t.Apply(func(a float64) float64 {
		return 1 - a
	})
	// part3 : log(1-y)
	part3 := y.Apply(func(a float64) float64 {
		return math.Log(1 - a)
	})
	// part4 : (1-t)*log(1-y))
	part4 := part2.Transpose().Multiply(part3)
	return part1.Apply2Matrix(part4, func(v1, v2 float64) float64 {
		return -(v1 + v2)
	}).Rows[0][0] / (float64(n))
}

func (b *BinaryCrossEntropy) BackWard() matrix.Matrix {
	// g = -(t/y - (1-t)/(1-y)) / N
	return b.InputT.Apply2Matrix(b.InputY, func(m, n float64) float64 {
		return -(m/n - (1-m)/(1-n)) / float64(b.InputT.RowSize)
	})
}

func (n Net) Init() {
	for _, layer := range n.Sequential {
		layer.Init()
	}
}

func (n Net) FeedFoward(input matrix.Matrix) matrix.Matrix {
	for _, layer := range n.Sequential {
		input = layer.FeedFoward(input)
	}
	return input
}

func (n Net) BackWard(gradiantNext matrix.Matrix) {
	for i := len(n.Sequential) - 1; i >= 0; i-- {
		gradiantNext = n.Sequential[i].BackWard(gradiantNext)
	}
}

func (n Net) Update(lr float64) {
	for _, layer := range n.Sequential {
		layer.Update(lr)
	}
}
