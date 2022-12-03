
namespace NeuralNetwork
{
    public class Sigmoid : IActivator
    {
        public double Activate(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }

        public double Dx(double x)
        {
            var sigmoid = this.Activate(x);
            return sigmoid * (1 - sigmoid);
        }
    }
}
