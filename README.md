# ChARM (Chinese Authorship Recognition Model)

This tool is designed to predict if a piece of traditional Tang poetry was written by a man or a woman.

## How to Access
- Hosted: https://charm-project.streamlit.app/
    - The app will deactivate due to inactivity, but a press of the button will awaken it in just a minute or two
- Local:
    - In the terminal, build the virtual environment:
        - `pipenv shell`
        - `pipenv install --ignore-pipfile`
    - Run the app:
        - `streamlit run main.py`

## Model Information
#### TF-IDF:
The term frequency-inverse document frequency model uses a support vector machine (SVM) to classify the vectors created using TF-IDF weights. These weights are calculated by dividing the count of each word in each document by the logarithm of the ratio of documents containing the word to the total number of documents.

##### Validation Results

- Accuracy Score: 82.43%
- Precision: 71.43%
- Recall: 31.35%
- F-Score: 43.48%

#### RNN:
The recurrent neural network model is built with tensorflow layers. There are three recurrent layers: a bidirectional simple RNN, a second simple RNN (not bidirectional), and finally a bidirectional long short-term memory (LSTM) layer. There are approximately 430,000 trainable parameters. The model is saved as a keras file in the source code for further inspection.

##### Validation Results

- Accuracy Score: 86.49%
- Precision: 75.00%
- Recall: 56.25%
- F-Score: 64.29%

## Training Information
#### The Data
The complete dataset included 317 poems by men and 85 poems by women. Punctuation is stripped from the poems, but an underscore is used to indicate a new line. When splitting the dataset into training and validation sets, the data was stratified, such that there was an approximately even ratio of poems by women to poems by men. The training set contained 328 poems (259 by men, 69 by women), and the validation set contained 74 poems (58 by men, 16 by women). The demo poems were not included in the training or validation sets.

#### Sources

The poems were sourced from around the web. The men's poems were scraped from the book 300 Tang Poems, alongside some additional poems collected by the Chinese Text Initiative. The women's poems were copy and pasted from journals and articles from across the internet.

#### Works Cited

“300 Tang Poems.” *Home of 300 Tang Poems*, Chinese Text Initiative. 

“Empress Wu Zetian Loves to Write Poems, These Ten Poems of Her Show the Spirit of the Emperor, and Reading Them Has a New Understanding of Her.” *iNews*. 

Jia, Jinhua. “The Yaochi Ji and Three Daoist Priestess-Poets in Tang China.” *Nan Nü*, vol. 13, no. 2, 2011, pp. 205–243. 

Kroll, Paul W. “The Life and Writings of Xu Hui (627-650), Worthy Consort, at the Early Tang Court.” *Asia Major*, vol. 22, no. 2, 2009, pp. 35–64. 

Lewis, Anna. “Rebel Within a Clause: Innovation in the Poetry of Xue Tao.” *New College of Florida*, 2013. 

Samei, Maija Bell. “Women and Poetry in the Tang Dynasty: A Traitor and a Murderess: The Poetic Nuns Li Ye and Yu Xuanji.” *How to Read Chinese Poetry Podcast*, Lingnan University, 4 Oct. 2022. 

Samei, Maija Bell. “Women and Poetry in the Tang Dynasty: Writing Women from the Inner Quarters to the Halls of Power: Shangguan Wan’er.” *How to Read Chinese Poetry Podcast*, Lingnan University, 20 Sept. 2022. 

Wu, Jie. “Vitality and cohesiveness in the poetry of Shangguan Wan’er (664—710).” *Tang Studies*, vol. 34, no. 1, 2016, pp. 40–72. 

Yu, Lu, and Tao Xue. *Readings of Chinese Poet Xue Tao*, University of Massachusetts Amherst, 2010. 

Zetian, Wu. “从驾幸少林寺.” *PoetryNook*. 
