# High-Cardinality-Cetegorical-Featue-Handling
E-mail : [yltsai0609@gmail.com](yltsai0609@gmail.com) <br>
**********************************************
## Introduction
類別特徵(nominal feature)，有的特徵會有非常多類別，我們稱之為高基數類別特徵(high cardinality nomial feature)，常見的包含(地區，行政區，ip位置，會員id，會員所屬校區，甚至是ubike在台北市的站點等)。<br>
在高基數類別特徵的預測中，由於各項特徵對預測目標(target)有不同的影響，但又並非是有序特徵(ordinal feature)一般有順序性，因此對於Tree-based model來說非常容易造成overfitting。
本篇實作了<br><br>
A Preprocessing Scheme for High-Cardinality Categorical
Attributes in Classification and Prediction Problems[[1]](#ref)<br>
Entity Embeddings of Categorical Variables[[2]](#ref)兩種解決high cardinality的encoding方式，並以Label Encoding, One-Hot Encoding作為benchmark進行比較，並且用直觀的方法[[1]](#ref)<br><br>
中所提到的Target Encoding(又稱mean encoding, likelihood encoding, impact encoding)其中的參數，你可以直接執行 main.py獲取結果，或是從display_notebook.ipynb閱讀實作的code。<br>

## Data

這次的示範資料集是從Kaggle上2013年的[Amazon員工訪問權限預測挑戰賽](https://www.kaggle.com/c/amazon-employee-access-challenge)中取得
這個資料集，該資料集收集了Amazon公司中各個員工針對每個資源(例如網頁的logging)的訪問紀錄，當員工屬於能夠取得訪問權限時，系統卻不給訪問，又要向上申請才能取得權限，一來一往浪費的非常多時間，因此這場比賽希望能夠建構模型，減少員工訪問權限所需的人工流程，我們取出5個特徵如下 :

* Feature (X)

> RESOURCE : 資源ID

> MGR_ID : 員工主管的ID 

> ROLE_FAMILY_DESC : 員工類別擴展描述 (例如 軟體工程的零售經理)

> ROLE_FAMILY : 員工類別 (例如 零售經理)

> ROLE_CODE : 員工角色編碼 (例如 經理)

* Target (Y)

> ACTION : 

 >> 1 : RESOURCE 訪問權限取得
 
 >> 0 : RESOURCE 禁止訪問
 
 各特徵基數值
 
 |feature|count of unqiues|
 |-------|----------------|
 |RESOURCE|6711|
 |MGR_ID|4062|
 |ROLE_FAMILY_DESC|2201|
 |ROLE_FAMILY|67|
 |ROLE_CODE|337|
 
 

### Target encoding
#### how it work?
Target encoding的中心思想為 :
將類別特徵轉換為數值特徵，使用該特徵中每個種類對於target的mean值:
例如特徵ROLE_FAMILY

|value|target|target encoding value|
|-----|------|---------------------|
|118424|1|1|
|22434|0|0.33|
|118424|1|1|
|22434|1|0.33|
|1855|0|0|
|118424|1|1|
|118424|1|1|
|118424|1|1|
|22434|0|0.33|

透過以上我們可以發現 ROLE_FAMILY 為 118424時 target 都會 = 1，22434則是一個為1, 一個為0，因此平均為0.5，
如此一來我們將類別特徵透過target值轉成數值型特徵。

#### estimated mean / overall mean

* overall mean : 在此例子中，如果我們完全不看ROLE_FAMILY的值，單純看有幾筆資料，幾個target = 1, 幾個 = 0
則我們可以得到共9個值, 6個target = 1, 3個target = 0， <b>因此在不考慮ROLE_FAMILY帶有的資訊的情況下</b>， overall_mean = 0.66
* estimated mean : 
我們可以從上述例子看到，ROLE_FAMILY中，118424共出現5次，22434出現3次，1855則出現1次，
<b>考慮ROLE_FAMILY帶有的情況下</b>

|value|mean of target|
|-----|--------------|
|118424|1|
|22434|0.33|
|1855|0|

* 該相信哪個?
我們可以從上述例子看到，ROLE_FAMILY中，118424共出現5次，22434出現3次，1855則出現1次，
我們無從判斷1855的target value是否是運氣成份使然, 還是真的是1，因此，當種類出現的次數(count越少)
我們越傾向不相信該mean值, 轉而相信全局平均值，顯然，我們可以透過種類出現的次數，來決定我們究竟該多相信estimated mean <br>
`smoothing_mean = smoothing_factor * estimated_mean + (1 - smoothing_factor) * overall_mean`

* smoothing_actor

論文中提到透過兩個參數來決定smoothing_factor

`smoothing_factor = 1 / (1 + np.exp(- (counts - min_samples_leaf) / smoothing_slope))`
這個函數長得很像Logistic Regression中的sigmoid函數, 我們可以針對其中幾個特例點

https://www.youtube.com/watch?v=irkV4sYExX4&fbclid=IwAR3nd7_anJmxs3Esa096nlEr3-DDLGMoH5wIZD8W4BXU7ErZnoSDSEwhNe8

### Embedding
#### how it work?

<h2 id=ref> Reference <h2>
[1] [A Preprocessing Scheme for High-Cardinality Categorical
Attributes in Classification and Prediction Problems](http://delivery.acm.org/10.1145/510000/507538/p27-micci-barreca.pdf?ip=180.217.111.180&id=507538&acc=ACTIVE%20SERVICE&key=CB7B71C8A2C31385%2E18DEC3E5D9CB506C%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1565526487_e186502b85ceb5db2c3bbc5efeb0c6e3)
[Entity Embeddings of Categorical Variables](https://arxiv.org/pdf/1604.06737.pdf)

