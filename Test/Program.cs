using NeuralNetwork;
using NeuralNetwork.Mnist;
using System.Drawing;
using System.Net.NetworkInformation;

namespace Test
{
    internal class Program
    {
        private static SimpleNeuralNetwork NN;
        private static int epochs = 0;
        private static string imageFile = @"..\..\..\Files\train-images.idx3-ubyte";
        private static string labelFile = @"..\..\..\Files\train-labels.idx1-ubyte";
        private static string imageFile1 = @"..\..\..\Files\t10k-images.idx3-ubyte";
        private static string labelFile1 = @"..\..\..\Files\t10k-labels.idx1-ubyte";
        private static string imagePath = @"..\..\..\Files\3.png";
        private static Dataset train;
        private static Dataset test;

        static void Main(string[] args)
        {
            //NN = new SimpleNeuralNetwork(new int[] { 28 * 28, 500, 150, 10 });
            NN = SimpleNeuralNetwork.Load(@"C:\Users\Cejurs\Weights.txt");
            train = new Dataset(MnistItem.LoadDataset(imageFile, labelFile).ToList());
            test = new Dataset(MnistItem.LoadDataset(imageFile1, labelFile1).ToList());

            //Learn();
            //TestMnist();
            Check();

        }
        private static void Check()
        {
            var image = new Bitmap(Image.FromFile(imagePath));
            var data = new List<byte>(784);
            for (int i = 0; i< image.Width; i++)
            {
                for (int j = 0; j<image.Height;j++)
                {
                    data.Add(image.GetPixel(j,i).R);
                }
            }
            var mnistItem = new MnistItem(data.ToArray(), 6);
            var dataset = new Dataset(new List<MnistItem>() { mnistItem});
            var output = NN.Predict(dataset.Inputs[0]);
            double max = -1;
            var answer = -1;
            for (int k = 0; k < 10; k++)
            {
                if (output[k] > max)
                {
                    max = output[k];
                    answer = k;
                }
            }
            Console.WriteLine($"Ответ нейросети - {answer} с вероятностью {max}");


        }
        private static void Learn()
        {
            do
            {
                epochs++;

                for (int i = 0; i < 60000; i++)
                {
                    NN.Learn(train.Inputs[i], train.Outputs[i]);

                    if (i % 100 == 0) Console.WriteLine("Train: - " + i);
                }

                TestMnist();

                Console.WriteLine("Epochs = " + epochs);
                Console.WriteLine("+++++++++++++++++++++++++++++");

            } while (epochs!=2);
            NN.Save(@"C:\Users\Cejurs\Weights.txt");
        }

        private static void TestMnist()
        {
            int countOfCorrectlyRecognized = 0;
            for (int i = 0; i < 10000; i++)
            {
                var output = NN.Predict(test.Inputs[i]);
                var answer = -1;
                var neuralAnswer = -1;
                double max = -1;
                for (int j = 0; j < 10; j++) 
                {
                    if (test.Outputs[i][j] == 1)
                    {
                        answer = j;
                        break;
                    }
                }
                for(int k = 0; k<10; k++)
                {
                    if (output[k]> max)
                    {
                        max = output[k];
                        neuralAnswer = k;
                    }
                }
                if (answer == neuralAnswer)
                {
                    countOfCorrectlyRecognized++;
                }
            }
            Console.WriteLine("Тестовая выборка мнист = {0} \n", ((double)countOfCorrectlyRecognized / 10000) * 100 + "%");
        }
    }
}