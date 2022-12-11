using System.Collections.Generic;
using System;
using NeuralNetwork.Mnist;

namespace NeuralNetwork
{
    public class Dataset
    {
        public List<double[]> Inputs { get; } // Лист входных сигналов
        public List<double[]> Outputs { get; } //Лист выходных
        public int Count => Inputs.Count;
        public Dataset(List<double[]> inputs, List<double[]> outputs)
        {
            if (inputs.Count != outputs.Count) throw new ArgumentException();
            Inputs = inputs;
            Outputs = outputs;
        }
        public Dataset(List<MnistItem> images) // Датасет по mnist
        {
            var inputs = new List<double[]>();
            var outputs = new List<double[]>();
            for (int i = 0; i < images.Count; i++)
            {
                inputs.Add(Normalize(images[i].Data)); // Входные значения от 0 до 1
                outputs.Add(CreateOutputArray(images[i].Label));
            }
            Inputs = inputs;
            Outputs = outputs;
        }

        private double[] Normalize(byte[] data)
        {
            var array = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                array[i] = data[i] / 255.0;
            }
            return array;
        }
        private double[] CreateOutputArray(byte digit)
        {
            var array = new double[10]; // массив чисел 0-9
            for (int i = 0; i < 10; i++)
            {
                if (digit == i) array[i] = 1;
                else array[i] = 0;
            }
            return array;
        }
    }
}
