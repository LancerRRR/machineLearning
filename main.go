package main

import (
	"encoding/csv"
	"fmt"
	nn "machineLearning/NN"
	"machineLearning/util"
	"os"
	"strconv"
)

func main() {
	file, _ := os.Open("05-dataset.tsv")
	defer file.Close()
	reader := csv.NewReader(file)
	reader.Comma = '\t'
	csvData, err := reader.ReadAll()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	X := [][]float64{}
	Y := [][]float64{}
	for _, each := range csvData {
		line := []float64{}
		for i, n := range each {
			if i > 0 {
				value, _ := strconv.ParseFloat(n, 10)
				line = append(line, value)
			} else {
				value, _ := strconv.ParseFloat(n, 10)
				Y = append(Y, []float64{value})
			}
		}
		X = append(X, line)

	}
	trainX := matrix.InitMatrix(X[:20000])
	trainY := matrix.InitMatrix(Y[:20000])
	testX := matrix.InitMatrix(X[20000:])
	testY := matrix.InitMatrix(Y[20000:])
	net := nn.Net{}
	layer1 := nn.Linear{}
	layer1.InSize = 7
	layer1.OutSize = 16
	net.Sequential = []nn.Layer{
		&layer1,
		&nn.Sigmoid{},
		&nn.Linear{InSize: 16, OutSize: 1},
		&nn.Sigmoid{},
	}
	lossFunc := nn.BinaryCrossEntropy{}
	net.Init()
	epochs := int(1000)
	for i := 0; i < epochs; i++ {
		predictY := net.FeedFoward(trainX)
		loss := lossFunc.Loss(predictY, trainY)
		g := lossFunc.BackWard()
		net.BackWard(g)
		net.Update(0.1)
		if i%50 == 0 {
			fmt.Println("loss: ", loss)
			predictY = predictY.Apply(func(a float64) float64 {
				if a > 0.5 {
					return 1
				} else {
					return 0
				}
			})
			predictY = predictY.Apply2Matrix(trainY, func(a, b float64) float64 {
				if a == b {
					return 1
				} else {
					return 0
				}
			})
			acc := matrix.Sum(predictY) / float64(predictY.RowSize)
			fmt.Println("Accuracy: ", acc)

		}

	}
	predictY := net.FeedFoward(testX)
	predictY = predictY.Apply(func(a float64) float64 {
		if a > 0.5 {
			return 1
		} else {
			return 0
		}
	})
	predictY = predictY.Apply2Matrix(testY, func(a, b float64) float64 {
		if a == b {
			return 1
		} else {
			return 0
		}
	})
	acc := matrix.Sum(predictY) / float64(predictY.RowSize)
	fmt.Println("Test ACC: ", acc)
}
