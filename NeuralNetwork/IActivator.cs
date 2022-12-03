namespace NeuralNetwork
{
    public interface IActivator
    {
        double Activate(double x);
        double Dx(double x);
    }
}