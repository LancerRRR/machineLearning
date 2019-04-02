package matrix

import (
	"fmt"
)

type Matrix struct {
	ColumnSize int
	RowSize    int
	Columns    [][]float64
	Rows       [][]float64
}

// init a matrix with a given data set
func InitMatrix(data [][]float64) Matrix {
	matrix := Matrix{}
	rowSize := len(data)
	if rowSize == 0 {
		panic("invalid format of input")
	}
	columnSize := len(data[0])
	if columnSize == 0 {
		panic("invalid format of input")
	}
	matrix.ColumnSize = columnSize
	matrix.RowSize = rowSize
	rows := [][]float64{}
	columns := [][]float64{}
	for i := 0; i < columnSize; i++ {
		column := []float64{}
		for _, row := range data {
			if i >= len(row) {
				panic("invalid format of input")
			}
			rows = append(rows, row)
			column = append(column, row[i])
		}
		columns = append(columns, column)
		if i > 0 {
			continue
		}
		matrix.Rows = rows
	}
	matrix.Columns = columns
	return matrix
}

func (m Matrix) String() string {
	res := ""
	for i, row := range m.Rows {
		res += fmt.Sprint(row)
		if i != m.RowSize-1 {
			res += "\n"
		}
	}
	return res
}

// output a transpose matrix of a given matrix
func (m Matrix) Transpose() Matrix {
	matrix := Matrix{}
	matrix.ColumnSize = m.RowSize
	matrix.RowSize = m.ColumnSize
	matrix.Columns = m.Rows
	matrix.Rows = m.Columns
	return matrix
}

// matrix multiplication
func (m Matrix) Multiply(n Matrix) Matrix {
	if m.ColumnSize != n.RowSize {
		panic("multiplication of " + fmt.Sprint(m.RowSize) + "X" + fmt.Sprint(m.ColumnSize) + " and " + fmt.Sprint(n.RowSize) + "X" + fmt.Sprint(m.ColumnSize) + " matrixes.")
	}
	matrix := Matrix{}
	matrix.RowSize = m.RowSize
	matrix.ColumnSize = n.ColumnSize
	rows := make([][]float64, matrix.RowSize)
	columns := make([][]float64, matrix.ColumnSize)
	for i, _ := range m.Rows {
		rows[i] = make([]float64, matrix.ColumnSize)
		for j, _ := range n.Columns {
			element := DotProduct(m.Rows[i], n.Columns[j])
			rows[i][j] = element
			if ok := columns[j]; ok == nil {
				columns[j] = make([]float64, matrix.RowSize)
			}
			columns[j][i] = element
		}
	}
	matrix.Columns = columns
	matrix.Rows = rows
	return matrix
}

// apply an operation to all elements in the matrix
func (m Matrix) Apply(f func(num float64) float64) Matrix {
	n := Matrix{}
	n.RowSize = m.RowSize
	n.ColumnSize = m.ColumnSize
	n.Rows = make([][]float64, n.RowSize)
	n.Columns = make([][]float64, n.ColumnSize)
	for i, row := range m.Rows {
		n.Rows[i] = make([]float64, n.ColumnSize)
		for j, _ := range row {
			if ok := n.Columns[j]; ok == nil {
				n.Columns[j] = make([]float64, n.RowSize)
			}
			result := f(m.Rows[i][j])
			n.Rows[i][j] = result
			n.Columns[j][i] = result
		}
	}
	return n
}

// apply operations between two matrixes
func (m Matrix) Apply2Matrix(n Matrix, f func(a, b float64) float64) Matrix {
	if m.RowSize != n.RowSize || m.ColumnSize != n.ColumnSize {
		panic("two matrixes must have same size")
	}
	resultM := Matrix{}
	resultM.RowSize = m.RowSize
	resultM.ColumnSize = m.ColumnSize
	resultM.Rows = make([][]float64, m.RowSize)
	resultM.Columns = make([][]float64, m.ColumnSize)
	for i, row := range m.Rows {
		resultM.Rows[i] = make([]float64, resultM.ColumnSize)
		for j, _ := range row {
			if ok := resultM.Columns[j]; ok == nil {
				resultM.Columns[j] = make([]float64, resultM.RowSize)
			}
			result := f(m.Rows[i][j], n.Rows[i][j])
			resultM.Rows[i][j] = result
			resultM.Columns[j][i] = result
		}
	}
	return resultM
}

func (m Matrix) SumAlongAxis(axis int) Matrix {
	n := Matrix{}
	if axis > 1 {
		panic("axis 2 is out of bounds for array of dimension 2")
	}
	if axis == 0 {
		n.RowSize = 1
		n.ColumnSize = m.ColumnSize
		n.Columns = make([][]float64, n.ColumnSize)
		n.Rows = make([][]float64, 1)
		for i, column := range m.Columns {
			res := sum(column)
			n.Columns[i] = append(n.Columns[i], res)
			n.Rows[0] = append(n.Rows[0], res)
		}
		return n
	} else {
		n.ColumnSize = 1
		n.RowSize = m.RowSize
		n.Rows = make([][]float64, n.RowSize)
		n.Columns = make([][]float64, 1)
		for i, row := range m.Rows {
			res := sum(row)
			n.Rows[i] = append(n.Rows[i], res)
			n.Columns[0] = append(n.Columns[0], res)
		}
		return n
	}
}

func (m Matrix) ApplyWithVector(n Matrix, f func(a, b float64) float64) Matrix {
	if n.ColumnSize != 1 && n.RowSize != 1 {
		panic("operation not allowed")
	}
	resultM := Matrix{}
	resultM.RowSize = m.RowSize
	resultM.ColumnSize = m.ColumnSize
	resultM.Columns = make([][]float64, resultM.ColumnSize)
	resultM.Rows = make([][]float64, resultM.RowSize)
	if n.ColumnSize == 1 && n.RowSize == 1 {
		return m.Apply(func(num float64) float64 {
			return f(num, n.Rows[0][0])
		})
	} else if n.ColumnSize == 1 {
		if n.RowSize != m.RowSize {
			panic("operation not allowed")
		}
		for i, row := range m.Rows {
			resultM.Rows[i] = make([]float64, resultM.ColumnSize)
			for j, _ := range row {
				if ok := resultM.Columns[j]; ok == nil {
					resultM.Columns[j] = make([]float64, n.RowSize)
				}
				result := f(m.Rows[i][j], n.Rows[i][0])
				resultM.Rows[i][j] = result
				resultM.Columns[j][i] = result
			}
		}
		return resultM
	} else {
		if n.ColumnSize != m.ColumnSize {
			panic("operation not allowed")
		}
		for i, row := range m.Rows {
			resultM.Rows[i] = make([]float64, resultM.ColumnSize)
			for j, _ := range row {
				if ok := resultM.Columns[j]; ok == nil {
					resultM.Columns[j] = make([]float64, resultM.RowSize)
				}
				result := f(m.Rows[i][j], n.Columns[j][0])
				resultM.Rows[i][j] = result
				resultM.Columns[j][i] = result
			}
		}
		return resultM
	}
}

func Sum(m Matrix) float64 {
	res := float64(0)
	for i, _ := range m.Rows {
		for j, _ := range m.Rows[i] {
			res += m.Rows[i][j]
		}
	}
	return res
}

func sum(vector []float64) float64 {
	res := float64(0)
	for _, n := range vector {
		res += n
	}
	return res
}

// compute dot product of two vectors
func DotProduct(vectorA, vectorB []float64) float64 {
	res := float64(0)
	for i, _ := range vectorA {
		res += vectorA[i] * vectorB[i]
	}
	return res
}

// func main() {
// 	a := [][]float64{[]float64{1, 2, 3}, []float64{4, 5, 6}}
// 	m := InitMatrix(a)
// 	b := [][]float64{[]float64{1}, []float64{2}}
// 	n := InitMatrix(b)
// 	fmt.Println("n: ", n)
// 	fmt.Println(m)
// 	fmt.Println(m.ApplyWithVector(n, func(a, b float64) float64 {
// 		return a + b
// 	}))

// }
