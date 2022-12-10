using System.IO;

namespace NeuralNetwork.Mnist
{
    public class MnistItem
    {
        public byte[] Data { get; set; } //Изображение 28 на 28
        public byte Label { get; set; } // 0-9

        public MnistItem(byte[] data, byte label)
        {
            Data = data;
            Label = label;
        }

        public static MnistItem[] LoadDataset(string imagefile, string labelfile)
        {
            BinaryReader images = new BinaryReader(new FileStream(imagefile, FileMode.Open));
            BinaryReader labels = new BinaryReader(new FileStream(labelfile, FileMode.Open));
            int magicNumber = images.ReadReverseInt32();
            int numberOfImages = images.ReadReverseInt32();
            int width = images.ReadReverseInt32();
            int height = images.ReadReverseInt32();

            int magicLabel = labels.ReadReverseInt32();
            int numberOfLabels = labels.ReadReverseInt32();

            MnistItem[] result = new MnistItem[numberOfImages];

            for (int i = 0; i < numberOfImages; i++)
            {
                var data = images.ReadBytes(width * height);
                byte label = labels.ReadByte(); // получаем маркеры
                var mnistImage = new MnistItem(data, label);
                result[i] = mnistImage;
            }
            return result;
        }
    }
}
