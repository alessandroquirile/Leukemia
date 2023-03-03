from utils import create_dataframe, show_image

if __name__ == '__main__':
    leukemia = "dataset/leukemia"
    hem = "dataset/healthy"

    df = create_dataframe(leukemia, hem)
    print(df)  # [0;7272) are tumoral cells

    show_image(df, 1000)
