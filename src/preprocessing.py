from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import matplotlib.pyplot as plt
import matplotlib as mpl
from bs4 import BeautifulSoup
from glob import glob
import pandas as pd
import numpy as np
import re
import os

ch_punct_list = ['，','。','？','•','；','！',',']
ch_punct_re = '[，。？•；！,]'

M_poems_path = os.path.join('data', 'M_poems.csv')
F_poems_path = os.path.join('data', 'F_poems.csv')

def save_M_poems() -> pd.DataFrame:
    # Scrape data from 300 Tang Poems
    url = "https://cti.lib.virginia.edu/tangeng.html"
    xpath = "/html/body"

    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 500)
    driver.get(url)

    transcript = wait.until(EC.visibility_of_element_located((By.XPATH, xpath)))
    html = transcript.get_attribute('innerHTML')

    driver.quit()

    # Use beautifulsoup to parse the html and save as csv
    bs = BeautifulSoup(html, "html.parser")
    mainbody = bs.find('notestmt')
    iter_ = mainbody.find_next('blockquote')

    M_poems = {'num': [], 
            'author': [], 
            'title': [], 
            'eng_title': [], 
            'poem':[]}

    while iter_ is not None:
        if iter_.text.isnumeric():
            M_poems['num'].append(int(iter_.text))
            
            iter_ = iter_.find_next('i')
            M_poems['title'].append(iter_.text)

            iter_ = iter_.find_next('blockquote')
            poem = iter_.text.strip()
            poem = '\n'.join([x.strip() for x in poem.split('\n') if x.strip() != ''])
            poem = re.sub(fr"({ch_punct_re}) ",r'\1\n', poem)
            M_poems['poem'].append(poem)

            iter_ = iter_.find_next('blockquote').find_next('blockquote').find_next('blockquote')
            M_poems['author'].append(iter_.text.strip())

            iter_ = iter_.find_next('blockquote')
            M_poems['eng_title'].append(iter_.text.strip())

        iter_ = iter_.find_next('blockquote')

    M_df = pd.DataFrame(M_poems)

    # Adjust numbers for author (to be consistent with the women's data)
    all_authors = set(M_df['author'])
    count = {auth:0 for auth in all_authors}
    new_num = []
    for auth in M_df['author']:
        count[auth] += 1
        new_num.append(count[auth])
    M_df['num'] = new_num

    M_df['gender'] = ['M'] * len(M_df)

    # Remove a poem with an unknown character and those selected for demo purposes
    rtitles = ['漁翁','月下獨酌','玉臺體']
    ridx = list(M_df[M_df['title'].isin(rtitles)].index)
    M_df = M_df.drop(ridx, axis=0).reset_index(drop=True)

    M_df.to_csv(M_poems_path, index=False)

    return M_df

def save_F_poems() -> pd.DataFrame:
    files = sorted(glob(os.path.join('data', 'raw_poems', 'F_*.txt')))
    F_poems = {'num': [], 
               'author': [], 
               'title': [], 
               'eng_title': [], 
               'poem':[]}

    for file in files:
        with open(file, 'r') as f:
            text = f.read()
        
        first_line, poems = text.split('\n\n', 1)

        spl = re.split(r'(\d+)\n', poems)
        spl.remove('')

        auth = first_line.split(', ')[0]
        
        for i in range(0, len(spl), 2):
            F_poems['author'].append(auth)
            F_poems['num'].append(spl[i])
            
            title, poem = spl[i+1].strip().split('\n\n')
            
            if title == 'Unknown':
                ch_title = 'Unknown'
                eng_title = 'Unknown'
            else:
                ch_title, eng_title = title.split(' ', 1)
                eng_title = eng_title[1:-1]

            F_poems['title'].append(ch_title)
            F_poems['eng_title'].append(eng_title)
            
            F_poems['poem'].append(poem.strip())

    F_df = pd.DataFrame(F_poems)

    F_df['gender'] = ['F'] * len(F_df)

    F_df.to_csv(F_poems_path, index=False)
    
    return F_df

def get_M_poems() -> pd.DataFrame:
    M_df = pd.read_csv(M_poems_path)
    reformat_poem(M_df)
    return M_df

def get_F_poems() -> pd.DataFrame:
    F_df = pd.read_csv(F_poems_path)
    reformat_poem(F_df)
    return F_df

def get_M_poems_short(M_df: pd.DataFrame = None) -> pd.DataFrame:
    # Filter the poems so they are at most 150 characters long
    if M_df is None:
        M_df = get_M_poems()
    
    return M_df[M_df['length'] <= 150].reset_index(drop=True)

def get_F_poems_short(F_df: pd.DataFrame = None) -> pd.DataFrame:
    # Filter the poems so they are at most 150 characters long
    if F_df is None:
        F_df = get_F_poems()
    
    return F_df[F_df['length'] <= 150].reset_index(drop=True)

def remove_pinyin(text: str, 
                  chr_per_line: int
                  ) -> str:
    ch_text = re.sub('[a-zāēīōūǖáéíóúǘǎěǐǒǔǚàèìòùǜü \n]','',text)
    # b = re.sub('\n','',b)

    ch_text = ('\n').join(re.findall(".{1,%d}" % (chr_per_line),ch_text))

    return ch_text

def reformat_poem(df: pd.DataFrame):
    no_punct = np.zeros(len(df), dtype=object)
    lengths = np.zeros(len(df), dtype=int)

    for i, poem in enumerate(df['poem']):
        # Remove punctuation
        temp = re.sub(fr'{ch_punct_re}', '', poem)

        # Replace new lines with an _
        temp = re.sub('\n', '_', temp)

        # Save length of punctuation-less poem
        lengths[i] = len(temp)

        # Add spaces between each character so they are processed as individual words
        temp = temp.replace('', ' ').strip()

        # Save new format
        no_punct[i] = temp

    df['poem_no_punct'] = no_punct
    df['length'] = lengths
    
    return

def see_dist_graphs():
    M_df = get_M_poems()
    F_df = get_F_poems()
    M_df_short = get_M_poems_short(M_df)
    F_df_short = get_F_poems_short(F_df)

    fig, axs = plt.subplots(3, 2, figsize=(17, 8))

    axs[0,0].set_title('Number of documents in the full dataset')
    bars1 = axs[0,0].bar(x=['M', 'F'], height=[len(M_df), len(F_df)])
    axs[0,0].bar_label(bars1)

    axs[0,1].set_title('Distribution of the lengths of each document')
    axs[0,1].boxplot(
        [M_df['length'], F_df['length']],
        labels=['M','F']
    )
    
    axs[1,0].set_title('Number of documents in the shortened dataset')
    bars3 = axs[1,0].bar(x=['M', 'F'], height=[len(M_df_short), len(F_df_short)])
    axs[1,0].bar_label(bars1)

    axs[1,1].set_title('Distribution of the lengths of each document')
    axs[1,1].boxplot(
        [M_df_short['length'], F_df_short['length']],
        labels=['M','F']
    )

    axs[2,0].set_title('Distribution of length of poems written by men')
    axs[2,0].hist(M_df_short['length'])

    axs[2,1].set_title('Distribution of length of poems written by women')
    axs[2,1].hist(F_df_short['length'])

    plt.show()

    return

def get_ttv_idxs(n: int, 
                 train_pct: float, 
                 val_pct: float, 
                 random_seed: int = None
                 ) -> tuple[list, list, list]:
    if random_seed is not None:
        np.random.seed(random_seed)

    train_size = int(n * train_pct)
    val_size = int(n * val_pct)

    idxs = list(range(n))
    np.random.shuffle(idxs)

    train_idxs = idxs[ : train_size]
    val_idxs = idxs[train_size : train_size + val_size]
    test_idxs = idxs[train_size + val_size : ]

    return train_idxs, val_idxs, test_idxs

def get_splits(df, 
               split_idxs
               ) -> list:
    splits = []
    for split in split_idxs:
        splits.append(df.loc[split,:])

    return splits

def train_test_val_split(M_df: pd.DataFrame,
                         F_df: pd.DataFrame, 
                         x_col: str, 
                         y_col: str, 
                         train_pct: float = 0.75, 
                         val_pct: float = 0.15,
                         random_seed: int = None):

    if (train_pct + val_pct) > 1:
        raise ValueError("Sum of percentages must be less than or equal to 1.")
    np.random.seed(1693)

    M_splits_idx = get_ttv_idxs(len(M_df), train_pct, val_pct, random_seed)
    F_splits_idx = get_ttv_idxs(len(F_df), train_pct, val_pct, random_seed)
    
    M_splits = get_splits(M_df, M_splits_idx)
    F_splits = get_splits(F_df, F_splits_idx)

    train = pd.concat([M_splits[0], F_splits[0]], ignore_index=True).sample(frac=1).reset_index(drop=True)
    val = pd.concat([M_splits[1], F_splits[1]], ignore_index=True).sample(frac=1).reset_index(drop=True)
    test = pd.concat([M_splits[2], F_splits[2]], ignore_index=True).sample(frac=1).reset_index(drop=True)

    return train[x_col], val[x_col], test[x_col], train[y_col], val[y_col], test[y_col]

