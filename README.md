# Hierarchical Recommender with Gradient Routing Algorithm

## Description
A recommender platform containing the implementation of the gradient routing algorithm for experiments on open-source datasets.

## ClearML
This project is based on ClearML - tool for managing ML pipelines (e.g. experiment tracking, data versioning etc.). To make it work ClearML server needs to be set up.
You can use [a web ClearML server](https://app.clear.ml/) or deploy [your own instance](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server/).
On client side, proper environment variables are required to connect and make use of ClearML. These variables are stored in .env (see .env.example).
For more details check [ClearML documentation](https://clear.ml/docs/latest/docs/).

## Installation
Clone the repository and install the required gradient-routing-recsys package in a prepared virtual environment:
```
# Clone project
git clone <repo_url>
cd gradient-routing-recsys

# Create conda environment
conda create -n <recsys_env_name> python=3.11.13
conda activate <recsys_env_name>

# Install
pip install -e .

# Deactivate environment before next steps
conda deactivate
```

## Datasets

### Download and Conversion

The datasets need to be downloaded and converted to the RecBole format using the [RecDatasets](https://github.com/RUCAIBox/RecDatasets) conversion tools. First, clone the RecDatasets repository and set up its environment:

```
# Clone RecDatasets project
git clone https://github.com/RUCAIBox/RecDatasets
cd RecDatasets/conversion_tools

# Create conda environment
conda create -n <dataset_env_name> python=3.11.13
conda activate <dataset_env_name>

# Install dependencies
pip install -r requirements.txt
```

Then, follow the dataset-specific instructions below.

#### MovieLens 32M
As this project leverages the [MovieLens 32M](https://grouplens.org/datasets/movielens/) dataset, which is not supported in RecBole by default,
you need to download, convert, and copy it to the `gradient-routing-recsys` repository first:
```
# Download the MovieLens dataset and extract the dataset file.
wget http://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip -d ml-32m

# Get the atomic files of MovieLens dataset (use the configuration for ml-20m, as ml-32m is not officially supported)
python run.py --dataset ml-20m \
--input_path ml-32m/ml-32m --output_path output_data/ml-32m \
--convert_inter --convert_item
mv output_data/ml-32m/ml-20m.inter output_data/ml-32m/ml-32m.inter
mv output_data/ml-32m/ml-20m.item output_data/ml-32m/ml-32m.item

# Remove quotation marks as they might cause errors during experiments
sed -i 's/"/ /g' output_data/ml-32m/ml-32m.item

# Copy the dataset to the gradient-routing-recsys repo
cp -r output_data/ml-32m/ ../../gradient-routing-recsys/dataset/ml-32m/
```

#### Yelp
First, you need to download the [Yelp](https://business.yelp.com/data/resources/open-dataset/) dataset and then:

```
unzip Yelp-JSON.zip

mkdir yelp_dataset

tar -zxvf 'Yelp JSON'/yelp_dataset.tar -C ./yelp_dataset

python run.py --dataset yelp \
--input_path yelp_dataset --output_path output_data/yelp \
--convert_inter --convert_item --convert_user

# Remove quotation marks as they might cause errors during experiments
sed -i 's/"/ /g' output_data/yelp/yelp.item

# Copy the dataset to the gradient-routing-recsys repo
cp -r output_data/yelp/ ../../gradient-routing-recsys/dataset/yelp/
```

#### Amazon Reviews'14 (Video Games)

```
mkdir Amazon_Video_Games
cd Amazon_Video_Games

curl -L -o ./ratings_Video_Games.csv https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv
curl -L -o ./meta_Video_Games.json.gz https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz
gunzip meta_Video_Games.json.gz

cd ..
python run.py --dataset amazon_video_games --input_path Amazon_Video_Games --output_path output_data/Amazon_Video_Games --convert_inter --convert_item

# Copy the dataset to the gradient-routing-recsys repo
cp -r output_data/Amazon_Video_Games/ ../../gradient-routing-recsys/dataset/Amazon_Video_Games/
```

#### Steam

```
wget http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
wget http://cseweb.ucsd.edu/~wckang/steam_games.json.gz

mkdir steam-data
gunzip -c steam_reviews.json.gz > steam-data/steam_reviews.json
gunzip -c steam_games.json.gz > steam-data/steam_games.json

python run.py --dataset steam --input_path steam-data --output_path output_data/steam --convert_inter
python run.py --dataset steam --input_path steam-data --output_path output_data/steam --convert_item

# Apply the necessary patches for experiments
sed -i '1s/id:token/product_id:token/' output_data/steam/steam.item
sed -i 's/[[]//g' output_data/steam/steam.item
sed -i 's/[]]//g' output_data/steam/steam.item
sed -i 's/"/ /g' output_data/steam/steam.item

cp -r output_data/steam/ ../../gradient-routing-recsys/dataset/steam/

# Deactivate environment before next steps
conda deactivate
```

### Preparation with User Preferences

After downloading and converting datasets using the RecBole tools, you need to run the preparation scripts to add user preference vectors. These scripts process the raw interaction data and add preference vectors based on user interaction history.

#### Available Preparation Scripts

| Script | Dataset | Output Columns |
|--------|---------|----------------|
| `jobs/prepare_ml32m_dataset.py` | MovieLens 32M | `user_pref_vector_implicit`, `user_pref_vector_explicit` |
| `jobs/prepare_yelp_dataset.py` | Yelp | `user_pref_vector_implicit`, `user_pref_vector_explicit` |
| `jobs/prepare_steam_dataset.py` | Steam | `user_pref_vector_implicit` |
| `jobs/prepare_amazon_reviews_dataset.py` | Amazon Reviews (Video Games) | `user_pref_vector_implicit`, `user_pref_vector_explicit` |

#### Running Preparation Scripts

Before running any preparation script, ensure the raw data files are in place:

```bash
# MovieLens 32M
python jobs/prepare_ml32m_dataset.py

# Yelp
python jobs/prepare_yelp_dataset.py

# Steam
python jobs/prepare_steam_dataset.py

# Amazon Reviews (Video Games)
python jobs/prepare_amazon_reviews_dataset.py
```

#### What the Scripts Do

All preparation scripts follow a similar approach:

1. **Load and merge data**: Read the interaction and item metadata files, then merge them
2. **Extract categories**: Parse item categories/genres from the item metadata
3. **Sort interactions**: Order interactions by user and timestamp to establish the interaction sequence
4. **Calculate preference vectors**:
   - **First interaction**: Assign a uniform distribution across all categories
   - **Subsequent interactions**: Assign a distribution based on cumulative category history
5. **Save processed data**: Overwrite the original interaction file with the added preference vectors

##### Preference Vector Modes

| Mode | Description | Threshold           |
|------|-------------|---------------------|
| **Implicit** | Uses all interactions regardless of rating | N/A                 |
| **Explicit** | Uses only interactions with rating ≥ 4.0 | `thr_explicit>=4.0` |

**Note**: The Steam dataset only uses implicit mode since it contains implicit feedback (playtime/binary interactions).

#### Output Format

The processed interaction files will have the following structure:

**MovieLens 32M:**
| user_id:token | item_id:token | rating:float | timestamp:float | user_pref_vector_implicit:float_seq | user_pref_vector_explicit:float_seq |
|---------------|---------------|--------------|-----------------|------------------------------------|-----------------------------------|
| 1 | 123 | 4.0 | 1234567890 | "0.0455 0.0909 ..." | "0.0455 0.0909 ..." |

**Yelp:**
| review_id:token | user_id:token | business_id:token | stars:float | useful:float | funny:float | cool:float | date:float | user_pref_vector_implicit:float_seq | user_pref_vector_explicit:float_seq |
|-----------------|---------------|-------------------|-------------|--------------|-------------|------------|------------|------------------------------------|-----------------------------------|

**Steam:**
| user_id:token | product_id:token | user_pref_vector_implicit:float_seq |
|---------------|------------------|------------------------------------|
| 1 | 123 | "0.0455 0.0909 ..." |

**Amazon Reviews (Video Games):**
| user_id:token | item_id:token | rating:float | timestamp:float | user_pref_vector_implicit:float_seq | user_pref_vector_explicit:float_seq |
|---------------|---------------|--------------|-----------------|------------------------------------|-----------------------------------|

Preference vectors are space-separated strings with category probabilities rounded to 4 decimal places.

#### Category Handling

| Dataset | Category Source | Category Count |
|---------|-----------------|----------------|
| MovieLens 32M | Movie genres (space-separated) | All genres (~19) |
| Yelp | Business categories (comma-separated) | Top 100 + "other" |
| Steam | Game genres (comma-separated) | All genres |
| Amazon Reviews | Product categories | Top 100 + "other" |

### Available Configurations

The following dataset configurations are available in `configs/hydra/dataset/`:

| Configuration | Dataset | Feedback Type | User Preferences |
|---------------|---------|---------------|------------------|
| `ml_32m_explicit_labeled` | MovieLens 32M | Explicit (ratings) | No |
| `ml_32m_explicit_labeled_user_prefs` | MovieLens 32M | Explicit (ratings) | Yes |
| `yelp_explicit_labeled` | Yelp | Explicit (ratings) | No |
| `yelp_explicit_labeled_user_prefs` | Yelp | Explicit (ratings) | Yes |
| `amazon_video_games_explicit_labeled` | Amazon Video Games | Explicit (ratings) | No |
| `amazon_video_games_explicit_labeled_user_prefs` | Amazon Video Games | Explicit (ratings) | Yes |
| `steam_implicit` | Steam | Implicit | No |
| `steam_implicit_user_prefs` | Steam | Implicit | Yes |

## Hyperparameter Tuning

Activate the environment:
```
conda activate <recsys_env_name>
```

To find the best hyperparameters for a given model, we can use the following script:
```
python jobs/tune_hyperparameter.py
```

The above is equivalent to:
```
python jobs/tune_hyperparameter.py \
    dataset=ml_32m_explicit_labeled \
    model=hierarchical_recommender \
    model.input_user_fields_rec_net=False \
    user_pref_net=wide_deep \
    recommender_net=wide_deep
```

You do not have to define `user_pref_net` and `recommender_net` (as they are not applicable) unless you use the `hierarchical_recommender` model, e.g.:
```
python jobs/tune_hyperparameter.py \
    dataset=ml_32m_explicit_labeled \
    model=wide_deep
```

The following properties can be customized:

| Property                         | Options                                                                                                                                                                                                                                                                                                                                        |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset                          | ml_32m_explicit_labeled<br/>ml_32m_explicit_labeled_user_prefs<br/>yelp_explicit_labeled<br/>yelp_explicit_labeled_user_prefs<br/>amazon_video_games_explicit_labeled<br/>amazon_video_games_explicit_labeled_user_prefs<br/>steam_implicit<br/>steam_implicit_user_prefs |
| model                            | hierarchical_recommender<br/>deepfm<br/>fignn<br/>lr<br/>wide_deep                                                                                                                                                                                                                                                                             |
| model.input_user_fields_rec_net  | True (f_rec(v^U_u, ufp, v^I_i))<br/>False (f_rec(ufp, i))                                                                                                                                                                                                                                                                                      |
| model.user_pref_activation       | softmax (default)<br/>sigmoid<br/>gumbel_softmax<br/>sparsemax<br/>relu<br/>silu<br/>tanh                                                                                                                                                                                                                                                      |
| model.gumbel_softmax_temperature | float, default=1.0. Only used when model.user_pref_activation=gumbel_softmax. τ<1: more discrete, τ=1: balanced, τ>1: more continuous                                                                                                                                                                                                          |
| user_pref_net                    | deepfm<br/>lr<br/>wide_deep                                                                                                                                                                                                                                                                                                                                        |
| recommender_net                  | deepfm<br/>fignn<br/>lr<br/>wide_deep                                                                                                                                                                                                                                                                                                          |
| gpu_id                           | int, default=0. The ID of the GPU to use for training                                                                                                                                                                                                                                                                                          |
| worker                           | int, default=0. The number of workers for data loading (0 = all processes)                                                                                                                                                                                                                                                                     |

If you want to tune a model or submodel that is not currently supported, prepare parameter settings for it in `./hparams` and add them to the dictionary in `jobs/tune_hyperparameter.py`.

**Note:** The specified GPU must be available on your system. The script will raise an error if the GPU ID is invalid. These parameters apply to both `jobs/tune_hyperparameter.py` and `jobs/experiment.py`.

## Experimentation

**Note:** The results reported in the paper were generated through hyperparameter tuning, so running `python jobs/experiment.py` with default hyperparameters may yield different results, as the defaults are not necessarily optimal. The optimal hyperparameters for each dataset are provided in `best_hparams.md` file. You can set specific hyperparameters in the configuration files for a single experiment.

Activate the environment:
```
conda activate <recsys_env_name>
```

Run an experiment:
```
python jobs/experiment.py
```

The above is equivalent to:
```
python jobs/experiment.py \
    dataset=ml_32m_explicit_labeled \
    model=hierarchical_recommender \
    model.input_user_fields_rec_net=False \
    user_pref_net=wide_deep \
    recommender_net=wide_deep
```

You do not have to define `user_pref_net` and `recommender_net` (as they are not applicable) unless you use the `hierarchical_recommender` model, e.g.:
```
python jobs/experiment.py \
    dataset=ml_32m_explicit_labeled \
    model=wide_deep
```

The following properties can be customized:

| Property                          | Options                                                                                                                                                                                                                                                                   |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset                           | ml_32m_explicit_labeled<br/>ml_32m_explicit_labeled_user_prefs<br/>yelp_explicit_labeled<br/>yelp_explicit_labeled_user_prefs<br/>amazon_video_games_explicit_labeled<br/>amazon_video_games_explicit_labeled_user_prefs<br/>steam_implicit<br/>steam_implicit_user_prefs |
| model                             | hierarchical_recommender<br/>afm<br/>autoint<br/>dcnv2<br/>deepfm<br/>ffm<br/>fignn<br/>fnn<br/>kd_dagfm<br/>lr<br/>pnn<br/>wide_deep<br/>xdeepfm                                                                                                                         |
| model.input_user_fields_rec_net   | True (f_rec(v^U_u, ufp, v^I_i))<br/>False (f_rec(ufp, i))                                                                                                                                                                                                                 |
| model.user_pref_activation        | softmax (default)<br/>sigmoid<br/>gumbel_softmax<br/>sparsemax<br/>relu<br/>silu<br/>tanh                                                                                                                                                                                 |
| model.gumbel_softmax_temperature  | float, default=1.0. Only used when model.user_pref_activation=gumbel_softmax. τ<1: more discrete, τ=1: balanced, τ>1: more continuous                                                                                                                                     |
| user_pref_net                     | afm<br/>autoint<br/>dcnv2<br/>deepfm<br/>ffm<br/>fignn<br/>fnn<br/>kd_dagfm<br/>lr<br/>pnn<br/>wide_deep<br/>xdeepfm                                                                                                                                                      |
| recommender_net                   | afm<br/>autoint<br/>dcnv2<br/>deepfm<br/>ffm<br/>fignn<br/>fnn<br/>kd_dagfm<br/>lr<br/>pnn<br/>wide_deep<br/>xdeepfm                                                                                                                                                      |
| gpu_id                            | int, default=0. The ID of the GPU to use for training                                                                                                                                                                                                                     |
| worker                            | int, default=0. The number of workers for data loading (0 = all processes)                                                                                                                                                                                                |
