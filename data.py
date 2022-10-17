import pickle

pickle_dir = 'C:\MAU_dataset\pickle\ '

with open(pickle_dir.strip() + 'finalDataKR.pickle', 'rb') as handle:
    finalData = pickle.load(handle)

with open(pickle_dir.strip() + 'finalDescriptionKR.pickle', 'rb') as handle:
    finalDescription = pickle.load(handle)

# Net national wealth to Net National Income Ratio (1995 ~ 2021)
wealth_ratio = [5.770171165, 5.975523472, 5.884808064, 6.399320602, 6.005833626,  \
                5.787640572, 5.750982761, 5.695618153, 6.018857956, 6.218288422,  \
                6.702404022, 7.152865887, 7.389669895, 7.569231033, 7.730065346,  \
                7.533388615, 7.565679073, 7.647249699, 7.637476921,7.681171417,   \
                7.638151646, 7.695356369, 7.775409698, 7.995022297,8.319354057, 8.814061165, 8.834832191]