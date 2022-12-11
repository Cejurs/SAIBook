# Лабораторная работа по дисциплине "Системы искуственного интелекта"
Задача обучить компудахтер распознавать цифры.
<br>
В ходе работы была одета, обута и обучена нейросеть имеющая 784 входных сигналов (Черно-белые изображения 28*28), 500 нейронов на 1 скрытом слое, 150 на втором скрытом слое и 10 на выходном для каждой цифры соответсвенно.
Для обучения нейросети использовался Mnist датасет, содержащий 60 тысяч изображений рукописных цифр размером 28*28 пикселей, а также меток, обозначающих какая цифра показана на изображении. Для проверки правильности работы нейронной сети mnist датасет содержит еще 10 тысяч изображении.
Ссылка на mnist датасет - http://yann.lecun.com/exdb/mnist/
<br>
На рисунке 1 показана точность распознавания из выборки 10 тысяч.
<p align="center"> <img src="https://github.com/Cejurs/SAIBook/blob/master/Test/Files/.png?raw=true" alt="Sublime's custom image"/> </p>
<p align="center"> Рисунок 1 </p>
<br>
Для тестирования нейронной сети на наших собственных изображениях подготовим их в photoshop( писать пользовательский интерфейс мы не будем, только консоль только хардкорд)<br>
Нарисованная в фотошопе цифра 5:
<img src="https://github.com/Cejurs/SAIBook/blob/master/Test/Files/5.png?raw=true" alt="Sublime's custom image"/>
Результат работы нейронной сети для цифры 5 показан на рисунке 2.
<br>
<p align="center"> <img src="https://github.com/Cejurs/SAIBook/blob/master/Test/Files/test5.png?raw=true" alt="Sublime's custom image"/> </p>
<p align="center"> Рисунок 2 </p>
<br>
Нарисованная в фотошопе цифра 3:
<img src="https://github.com/Cejurs/SAIBook/blob/master/Test/Files/3.png?raw=true" alt="Sublime's custom image"/>
Результат работы нейронной сети для цифры 3 показан на рисунке 3.
<br>
<p align="center"> <img src="https://github.com/Cejurs/SAIBook/blob/master/Test/Files/test3.png?raw=true" alt="Sublime's custom image"/> </p>
<p align="center"> Рисунок 3 </p>
<br>
