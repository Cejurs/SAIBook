using System;
using System.Linq;

namespace NeuralNetwork
{
    public class Neuron
    {
        internal double[] weights;
        internal double bias;
        private double error;
        private int numberOfSynapses;
        public Neuron(int numberOfSynapses)
        {
            this.numberOfSynapses = numberOfSynapses;
            weights = new double[numberOfSynapses];
            this.bias = 1;
            RandomizeWeights();
        }


        internal double GetOutput(double[] input)
        {
            double power = 0;
            for (int r = 0; r < numberOfSynapses; r++)
            {
                power += weights[r] * input[r];
            }
            power += bias * 1;
            return 1 / (1 + Math.Pow(Math.E, -1 * power));
        }
        internal void UpdateWeights(double[] output)
        {
            double epsilone = 0.5;

            for (int r = 0; r < numberOfSynapses; r++)
            {
                weights[r] += epsilone * error * output[r];
            }
            bias += epsilone * error * 1;
        }
        internal void UpdateErrorOutputLayer(double target, double output)
        {
            error = (target - output) * output * (1 - output);
        }

        internal void UpdateError(Neuron[] neurons, double output, int magicNumber)
        {
            double summ = 0;

            for (int i = 0; i < neurons.Count(); i++)
            {
                summ += neurons[i].weights[magicNumber] * neurons[i].error;
            }
            error = output * (1 - output) * summ;
        }

        private void RandomizeWeights()
        {
            var rand = new Random();
            for (int r = 0; r < numberOfSynapses; r++)
            {
                weights[r] = (double)rand.Next(-10000, 10001) / 20000; // random [-0.5; 0.5]
            }
        }
    }
}
