import pandas as pd
import matplotlib.pyplot as plt
import regressor as reg

def show_some(data: pd.DataFrame):
    fig = plt.figure(figsize=(25,25))
    for i, (_, row) in enumerate(data.sample(n=min(9,len(data))).iterrows()):
        fig.add_subplot(3,3,i+1, title=f'{row["sample"]}-{row["scan"]} y:{row["y"]} z:{row["zoom"]}')
        img = cv2.imread(row['filename'])
        img = reg.prep_img(img)
        plt.imshow(img)
        plt.scatter(img.shape[1]/2,img.shape[0]/2, c='red', marker='x')
