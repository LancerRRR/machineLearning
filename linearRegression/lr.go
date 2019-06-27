package lr

import (
	"github.com/gonum/matrix/mat64"
)

type LinearRegression struct {
	Weights *mat64.Dense
}

func Out(x, w mat64.Matrix) *mat64.Dense {
	n, _ := x.Dims()
	y := mat64.NewDense(n, 1, nil)
	y.Mul(x, w)
	return y
}

func (lr *LinearRegression) Fit(x, t mat64.Matrix, l float64) {
	n, d := x.Dims()
	lr.Weights = mat64.NewDense(d, 1, nil)
	a := mat64.NewDense(d, d, nil)
	a.Mul(x.T(), x)
	eye := []float64{}
	for i := 0; i < d*d; i++ {
		eye = append(eye, l)
	}
	eyeMatrix := mat64.NewDense(d, d, eye)
	a.Add(a, eyeMatrix)
	a.Inverse(a)
	b := mat64.NewDense(d, n, nil)
	b.Mul(a, x.T())
	lr.Weights.Mul(b, t)
}

func (lr *LinearRegression) Predict(x mat64.Matrix) mat64.Matrix {
	return Out(x, lr.Weights)
}
