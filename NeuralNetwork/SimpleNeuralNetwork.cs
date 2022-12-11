using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace NeuralNetwork
{
    public class SimpleNeuralNetwork
    {
        public Neuron[][] Neurons;

        public SimpleNeuralNetwork(params int[] neuronsCountInLayers)
        {
            Neurons = new Neuron[neuronsCountInLayers.Length][];

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron[neuronsCountInLayers[i]];
                if (i != 0)
                {
                    for (int j = 0; j < neuronsCountInLayers[i]; j++)
                    {
                        Neurons[i][j] = new Neuron(neuronsCountInLayers[i - 1]);
                    }
                }
            }
        }
        private double[][] FeedForward(double[] input)
        {
            double[][] output = new double[Neurons.Length][];

            //Выход первого слоя
            output[1] = new double[Neurons[1].Count()];
            for (int j = 0; j < Neurons[1].Count(); j++)
            {
                output[1][j] = Neurons[1][j].GetOutput(input);
            }

            //Выходы оставшихся слоев
            for (int i = 2; i < output.Length; i++)
            {
                output[i] = new double[Neurons[i].Count()];
                {
                    for (int j = 0; j < Neurons[i].Count(); j++)
                    {
                        output[i][j] = Neurons[i][j].GetOutput(output[i - 1]);
                    }
                }
            }

            return output;
        }
        public double[] Predict(double[] input)
        {
            double[][] output = FeedForward(input);
            return output[Neurons.Count() -1];
        }
        public bool Learn(double[] input, double[] correctOutput)
        {
            double[][] output = FeedForward(input);
            int countlearning = 0;
            if (CompareVectors(correctOutput, output[Neurons.Count() - 1]))
            {
                return true;
            }

            while (!CompareVectors(correctOutput, output[Neurons.Count() - 1]))
            {
                countlearning++;
                if (countlearning > 1000)
                {
                    break;
                }

                for (int i = 0; i < output[Neurons.Count() - 1].Count(); i++)
                {
                    Neurons[Neurons.Count() - 1][i].UpdateErrorOutputLayer(correctOutput[i], output[Neurons.Count() - 1][i]);
                }

                for (int i = output.Count() - 2; i >= 1; i--)
                {
                    for (int j = 0; j < output[i].Count(); j++)
                    {
                        Neurons[i][j].UpdateError(Neurons[i + 1], output[i][j], j);
                    }
                }

                for (int i = Neurons.Count() - 1; i >= 1; i--)
                {
                    for (int j = 0; j < Neurons[i].Count(); j++)
                    {
                        if (i - 1 != 0)
                        {
                            Neurons[i][j].UpdateWeights(output[i - 1]);
                        }
                        else if (i - 1 == 0)
                        {
                            Neurons[i][j].UpdateWeights(input);
                        }
                    }
                }
                output = FeedForward(input);
            }

            return false;
        }

        private bool CompareVectors(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                return false;

            for (int i = 0; i < a.Length; i++)
                if ((a[i] == 0 && b[i] > 0.1) || (a[i] == 1 && b[i] < 0.9))
                    return false;

            return true;
        }

        public void Save(string path)
        {
            if (File.Exists(path) == false)
            {
                File.Create(path).Close();
            }
            using(var stream = new StreamWriter(path, false))
            {
                var neuronsPerLayer = new int[Neurons.Count()];
                for(int i = 0; i< Neurons.Count(); i++)
                {
                    neuronsPerLayer[i] = Neurons[i].Count();
                }
                var line = JsonSerializer.Serialize(neuronsPerLayer);
                stream.WriteLine(line);
                for(int i= 1; i < Neurons.Count(); i++)
                {
                    for (int j = 0; j < Neurons[i].Count(); j++)
                    {
                        line = JsonSerializer.Serialize(Neurons[i][j].weights);
                        stream.WriteLine(line);
                        line = Neurons[i][j].bias.ToString();
                        stream.WriteLine(line);
                    }
                }
            }
        }
        public static SimpleNeuralNetwork Load(string path)
        {
            if (File.Exists(path) == false) throw new FileNotFoundException();

            using(var stream = new StreamReader(path))
            {
                var line = stream.ReadLine();
                int[] neuronsPerLayer = JsonSerializer.Deserialize<int[]>(line);
                var neuralNetwork = new SimpleNeuralNetwork(neuronsPerLayer);
                for(int i = 1; i<neuronsPerLayer.Length; i++)
                {
                    for(int j =0; j < neuronsPerLayer[i]; j++)
                    {
                        line = stream.ReadLine();
                        double[] weights = JsonSerializer.Deserialize<double[]>(line);
                        double bias = double.Parse(stream.ReadLine());
                        neuralNetwork.Neurons[i][j].weights = weights;
                        neuralNetwork.Neurons[i][j].bias = bias;
                    }
                }
                return neuralNetwork;
            }
        }
    }
}
