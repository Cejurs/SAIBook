using System.IO;
using System;

namespace NeuralNetwork.Mnist
{
    public static class Extension
    {
        public static int ReadReverseInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
