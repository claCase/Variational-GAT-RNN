# %%
from typing import List, Dict
import teneto
import numpy as np
import tensorflow as tf
import pandas as pd
import networkx as nx
import os
import pickle as pkl
from tqdm.contrib import tzip
import tqdm
from src.CONSTANTS import NA_VALUE, EXCLUDE_COUNTRIES


class BaciDataLoader:
    def __init__(self):
        self.raw_data = pd.DataFrame(
            {
                "t": pd.Series(dtype=int),
                "i": pd.Series(dtype=int),
                "j": pd.Series(dtype=int),
                "k": pd.Series(dtype=str),
                "v": pd.Series(dtype=np.float32),
                "q": pd.Series(dtype=np.float32),
            }
        )
        self.country2idx_mapping = dict()
        self.idx2country_mapping = dict()
        self.product2idx_mapping = dict()
        self.idx2product_mapping = dict()
        self.idx2year_mapping = dict()
        self.year2idx_mapping = dict()
        self.country_codes = pd.read_csv("./data/country_codes_V202301.csv")

    def from_csv_path(
        self,
        path,
        start: int = 1995,
        end: int = 2021,
        save_path=None,
    ) -> None:
        data = []
        try:
            root, dirs, files = next(os.walk(path))
            for file in files:
                file_splits = file.split("_")
                is_baci = file_splits[0] == "BACI"
                if is_baci:
                    year: int = int(file_splits[2][1:])
                    if int(year) > end:
                        break
                    if start <= int(year) <= end:
                        print(f"loading file {file}")
                        _data = pd.read_csv(
                            os.path.join(root, file),
                            converters={
                                "t": int,
                                "i": int,
                                "j": int,
                                "k": str,
                                "v": np.float32,
                                "q": str,
                            },
                        )
                        data.append(_data)
            complete_data = pd.concat(data, ignore_index=True)
            self.raw_data = pd.concat([self.raw_data, complete_data], ignore_index=True)
            self.raw_data["q"] = self.raw_data["q"].replace(NA_VALUE, np.nan)
            self.raw_data["q"] = self.raw_data["q"].astype(np.float32)

            self.idx2year_mapping[0] = start
            self.year2idx_mapping[start] = 0
            if start != end:
                for i in range(end - start):
                    self.year2idx_mapping[start + i + 1] = i
                    self.idx2year_mapping[i] = start + i + 1
            products = complete_data["k"].unique()
            countries_i = complete_data["i"]
            countries_i = set(countries_i.unique())
            countries_j = complete_data["j"]
            countries_j = set(countries_j.unique())
            countries = countries_i.union(countries_j)
            products.sort()

            print("Building Products Mapping...")
            self.product2idx_mapping, self.idx2product_mapping = self.build_mapping(
                products
            )

            if save_path is not None:
                if not os.path.exists(save_path):
                    print(f"Creating path {save_path}")
                    os.makedirs(save_path)
                with open(os.path.join(save_path, "products_to_idx.pkl"), "wb") as file:
                    pkl.dump(baci_prod2idx, file)
                with open(os.path.join(save_path, "idx_to_products.pkl"), "wb") as file:
                    pkl.dump(baci_prod2idx, file)

            print("Building Countries Mapping...")
            self.country2idx_mapping, self.idx2country_mapping = self.build_mapping(
                countries
            )
            if save_path is not None:
                with open(
                    os.path.join(save_path, "countries_to_idx.pkl"), "wb"
                ) as file:
                    pkl.dump(baci_country2idx, file)
                with open(
                    os.path.join(save_path, "idx_to_countries.pkl"), "wb"
                ) as file:
                    pkl.dump(baci_idx2country, file)
        except Exception as e:
            raise e

    @staticmethod
    def build_mapping(values):
        idx = np.arange(len(values))
        values2idx = {}
        idx2values = {}
        for k, v in tzip(values, idx):
            values2idx[k] = v
            idx2values[v] = k
        return values2idx, idx2values

    def fill_missing_with_avg(self, column: str = "q"):
        col_idx = self.raw_data.columns.get_loc(column)
        # Check for null values
        naidx = (self.raw_data.loc[pd.isna(self.raw_data[column])]).index
        # Get null values groups
        raw_reindexed = self.raw_data.iloc[naidx].set_index(["t", "i", "k"])[column]
        namidx = raw_reindexed.index.unique()
        # Average null values group value, if it's still na then substitute with 0.
        mean_na = raw_reindexed.loc[namidx].groupby(["t", "i", "k"]).mean().fillna(0.0)
        # Assign average to grouped dataFrame
        raw_reindexed.loc[namidx, column] = mean_na.loc[namidx]
        # Assign average to original data
        self.raw_data.iloc[naidx, col_idx] = raw_reindexed.reset_index()
        # Clean garbage
        del raw_reindexed
        del mean_na
        del naidx
        del namidx

    def reduce_product_detail(self, detail: int):
        print("Reducing Products...")
        if detail > 6:
            raise ValueError("Detail cannot be greater than 6")
        elif detail == 6:
            print("Skipping reducing dataset")
            return
        reduced = self.raw_data["k"].astype(str).str[:detail]
        self.raw_data["kk"] = reduced
        self.raw_data.drop("k", axis=1, inplace=True)
        self.raw_data = self.raw_data.groupby(["t", "i", "j", "kk"]).sum().reset_index()
        self.raw_data.rename(columns={"kk": "k"}, inplace=True)
        products = self.raw_data["k"].unique()
        self.product2idx_mapping, self.idx2product_mapping = self.build_mapping(
            products
        )

    def remove_countries(self, countries: List[str] = None):
        print("Removing Countries...")
        if countries is None:
            countries = EXCLUDE_COUNTRIES
        remaining = ~self.country_codes["country_name_abbreviation"].isin(countries)
        remaining_countries = self.country_codes[remaining]["country_code"]
        self.raw_data = self.raw_data[
            self.raw_data["i"].isin(remaining_countries)
            & self.raw_data["j"].isin(remaining_countries)
        ].reset_index(drop=True)
        left_countries = set(
            [*self.raw_data["i"].to_list(), *self.raw_data["j"].to_list()]
        )
        self.country2idx_mapping, self.idx2country_mapping = self.build_mapping(
            left_countries
        )

    def to_tf_sparse(self) -> tf.SparseTensor:
        print("Building Sparse Index List")
        raw_indices = self.raw_data[["t", "i", "j", "k"]]
        max_prod = max(self.idx2product_mapping.keys())
        max_cty = max(self.idx2country_mapping.keys())
        max_t = max(self.idx2year_mapping.keys())
        price = self.raw_data["v"]
        quantity = self.raw_data["q"]
        indices_price = np.empty(shape=(len(raw_indices), 5))
        indices_quantity = np.empty(shape=(len(raw_indices), 5))

        for raw in tqdm.tqdm(raw_indices.iterrows(), total=len(raw_indices)):
            idx = raw[0]
            t, i, j, k = raw[1].to_list()
            t = self.year2idx_mapping[t]
            i = self.country2idx_mapping[i]
            j = self.country2idx_mapping[j]
            k = self.product2idx_mapping[k]
            indices_price[idx, :] = np.asarray((t, i, j, k, 0))
            indices_quantity[idx, :] = np.asarray((t, i, j, k, 1))

        spt = tf.SparseTensor(
            indices=np.concatenate([indices_price, indices_quantity], 0),
            values=np.concatenate([price, quantity], 0),
            dense_shape=(max_t + 1, max_cty + 1, max_cty + 1, max_prod + 1, 2),
        )
        return spt

    def to_networkx(self):
        return {
            t: nx.from_pandas_edgelist(
                df=self.raw_data[self.raw_data.t == t],
                source="i",
                target="j",
                edge_attr=["q", "v"],
                edge_key="k",
                create_using=nx.MultiDiGraph(),
            )
            for t in self.raw_data.t.unique()
        }

    def to_rugged(self):
        raise NotImplemented

    def save(self, path):
        if not os.path.exists(path):
            print(f"making path: {path}")
            os.makedirs(path)
        with open(os.path.join(path, "year2idx_mapping.pkl"), "wb") as file:
            pkl.dump(self.year2idx_mapping, file)

        with open(os.path.join(path, "country2idx_mapping.pkl"), "wb") as file:
            pkl.dump(self.country2idx_mapping, file)

        with open(os.path.join(path, "product2idx_mapping.pkl"), "wb") as file:
            pkl.dump(self.product2idx_mapping, file)

        with open(os.path.join(path, "idx2year_mapping.pkl"), "wb") as file:
            pkl.dump(self.idx2year_mapping, file)

        with open(os.path.join(path, "idx2country_mapping.pkl"), "wb") as file:
            pkl.dump(self.idx2country_mapping, file)

        with open(os.path.join(path, "idx2product_mapping.pkl"), "wb") as file:
            pkl.dump(self.idx2product_mapping, file)

        self.raw_data.to_csv(os.path.join(path, "raw_data.csv"))

    def load(self, path):
        if os.path.exists(path):
            print(f"Loading Data from {path}")
        else:
            raise ValueError(f"Path {path} does not exist")

        with open(os.path.join(path, "year2idx_mapping.pkl"), "rb") as file:
            self.year2idx_mapping = pkl.load(file)

        with open(os.path.join(path, "country2idx_mapping.pkl"), "rb") as file:
            self.country2idx_mapping = pkl.load(file)

        with open(os.path.join(path, "product2idx_mapping.pkl"), "rb") as file:
            self.product2idx_mapping = pkl.load(file)

        with open(os.path.join(path, "idx2year_mapping.pkl"), "rb") as file:
            self.idx2year_mapping = pkl.load(file)

        with open(os.path.join(path, "idx2country_mapping.pkl"), "rb") as file:
            self.idx2country_mapping = pkl.load(file)

        with open(os.path.join(path, "idx2product_mapping.pkl"), "rb") as file:
            self.idx2product_mapping = pkl.load(file)

        self.raw_data = pd.read_csv(os.path.join(path, "raw_data.csv"))
