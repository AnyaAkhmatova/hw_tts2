# TTS project (part 2)

Проект - домашнее задание hw_tts2 по курсу dla. Предназначен для воспроизведения вокодера Hi-FiGAN. 

____

За основу репозитория взят темплейт https://github.com/WrathOfGrapes/asr_project_template.git. Структура проекта изменена, большинство базовых классов были удалены, некоторые - перенесены в директории с классами-наследниками. Некоторые модули были удалены, некоторые - добавлены. Немного изменилась структура конфигов. В целом, просто адаптировали проект под реализацию Hi-FiGAN. 

Код для обучения модели - в hw_tts.

В папке inference содержится конфиг (config.json) и данные (wav-файлы и их транскрипты inference.txt) для инференса. 

train.py был подкорректирован под задачу. inference.py был написан для генерации аудио по мел-спектрограммам, полученным после обработки тестовых аудио классом MelSpectrogram (взят из семинарского ноутбука/ текста задания).

Датасет скачивается отсюда: https://keithito.com/LJ-Speech-Dataset/. Папка с финальными моделями ./final_model/ скачивается с google drive. Скрипты для скачивания приведены ниже.

Dockerfile не валиден, необходимые пакеты устанавливаются с помощью requirements.txt.

inference.ipynb содержит пример запуска инференса. В папке final_model хранятся результаты 3 экспериментов: exp4, exp7, exp8. В каждой поддиректории хранится конфиг, лог обучения, финальные веса модели model_best.pth и ноутбук с запуском обучения.

____

Устанавливать библиотеки нужно с помощью requirements.txt. Dockerfile невалидный.

Guide по установке:
```
git clone https://github.com/AnyaAkhmatova/hw_tts2.git
```
Из директории ./hw_tts2/ (устанавливаем библиотеки и нашу маленькую библиотечку, скачиваем датасет, финальные модели (результаты экспериментов exp4, exp7, exp8, логи, конфиги, чекпоинты, ноутбуки из kaggle)):

```
pip install -r requirements.txt
pip install .

wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2
mkdir ./data
mv ./LJSpeech-1.1 ./data/.

gdown https://drive.google.com/u/0/uc?id=1tU_JORVQmb7ek7SsZCZf3PimEXWWW3a3
unzip final_model.zip
rm -rf final_model.zip
```

Wandb:

```
import wandb

wandb.login()
```

Запуск train:

```
!python3 train.py -c ./hw_tts/configs/exp8.json
```

Запуск inference:

```
!python3 inference.py -c ./inference/config.json -r ./final_model/exp8/model_best.pth
```

Комментарий: обучение запускалось в kaggle, инференс - в google colab.

____

W&B Report: https://wandb.ai/team-from-wonderland/tts2_project/reports/HW_TTS2--Vmlldzo2MTY0MTMz?accessToken=86w0zpmor3g8zobovsbwzmj0q379nufyuf1fatks7noe4dlf6pzt9y0sb4kojfm2.

____

Бонусов нет.

____


Выполнила Ахматова Аня, группа 201.

