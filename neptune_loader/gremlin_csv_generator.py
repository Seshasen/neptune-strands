import os
import json
import sys
import uuid
import pandas as pd
from icecream import ic
from neptune_loader.wrapper_functions import on_failure
import boto3
from langchain.embeddings import BedrockEmbeddings
import pandas as pd

# Ensure nodes and edges directories exist
os.makedirs("nodes", exist_ok=True)
os.makedirs("edges", exist_ok=True)


def node_filename(root_name: str, label: str, file_name: str = None) -> str:
    """For a parm label name, return the name of the Gremlin CSV filename for it."""
    if file_name:
        label = file_name
    return ("nodes/" + root_name + "_" + label + ".csv").replace(" ", "_").lower()


def get_uuid():
    return uuid.uuid4()


def get_dfs(root_name: str, edge) -> tuple:
    from_df = pd.read_csv(node_filename(root_name, edge.get("from_label"))).apply(
        lambda x: x.astype(str)
    )

    to_df = pd.read_csv(node_filename(root_name, edge.get("to_label"))).apply(
        lambda x: x.astype(str)
    )
    return from_df, to_df


@on_failure("Failure in white space remover!!")
def whitespace_remover(dataframe):
    """Removes leading and trailing whitespace from object type columns in a pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The input pandas DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with whitespace removed from object type columns.
    """

    # iterating over the columns
    for i in dataframe.columns:
        # checking datatype of each columns
        if dataframe[i].dtype == "object":
            # applying strip function on column
            dataframe[i] = dataframe[i].apply(lambda x: str(x).strip())
        else:
            # if condn. is False then it will do nothing.
            pass
    return dataframe


def remove_nulls(df, column):
    if column in df.columns:
        df = df.dropna(subset=[column])
        df = df[df[column] != "nan"]

    return df


@on_failure("Failure in Node creation for Operational Instruction data.")
def get_graph_nodes(df, node_dtl):
    label = node_dtl.get("label")

    cols = node_dtl.get("node_cols")
    ic(cols)

    filter_query = node_dtl.get("filter")

    ic(filter_query)

    if filter_query:
        df = df.query(filter_query)

    if cols:
        nodes_df = df[cols].drop_duplicates()
        nodes_df.columns = nodes_df.columns.str.replace(" ", "")
        nodes_df["~label"] = label
    else:
        nodes_df = pd.DataFrame([label], columns=["~label"])


    properties = node_dtl.get("properties")
    for key, value in properties.items():
        if key.find(":") < 0:
            key = key + ":string"
        if value in df.columns:
            nodes_df[key] = df[value]
        else:
            nodes_df[key] = value

    nodes_df = nodes_df.reset_index(drop=True)

    nodes_df["~id"] = [get_uuid() for _ in range(len(nodes_df.index))]

    ic(nodes_df.columns)

    # # Smart Symptoms car mechanics for Q4
    # nodes_df = remove_nulls(nodes_df, "question:string")
    #
    # nodes_df = remove_nulls(nodes_df, "answer:string")

    return nodes_df


def get_connect_edges(df, edge_dtl, from_df, to_df):
    def _get_label(label: str, upd_df: pd.DataFrame):
        if label in upd_df.columns:
            return upd_df[label]
        elif label + ":string" in upd_df.columns:
            return upd_df[label + ":string"]
        else:
            return label

    label = edge_dtl.get("label")
    to_df = to_df.replace("nan", "N/A",regex=False)
    from_df = from_df.replace("nan", "N/A", regex=False)

    from_cols = edge_dtl.get("from_cols")

    to_cols = edge_dtl.get("to_cols")

    properties = edge_dtl.get("properties")

    filter_query = edge_dtl.get("filter")

    ic(filter_query)

    upd_df = df.copy()

    if filter_query:
        upd_df = df.query(filter_query)
        upd_df = upd_df.reset_index(drop=True)

    if from_cols == 1 and to_cols == 1:
        root_edge_df = from_df.assign(key=1).merge(
            to_df.assign(key=1), on="key", how="outer"
        )

        root_edge_df["~label"] = _get_label(label, upd_df)

        root_edge_df["~from"] = root_edge_df["~id_x"]
        root_edge_df["~to"] = root_edge_df["~id_y"]
        edge_df = root_edge_df.copy()

        edge_df = edge_df.iloc[:, -3:]

    else:
        edge_df = upd_df.merge(
            from_df, left_on=from_cols, right_on=from_cols, how="left"
        ).rename(columns={"~id": "from_cols_id"})
        edge_df = edge_df.merge(
            to_df,
            left_on=to_cols,
            right_on=to_cols,
            how="left",
        ).rename(columns={"~id": "to_cols_id"})

        edge_df["~label"] = _get_label(label, upd_df)
        edge_df["~from"] = edge_df["from_cols_id"]
        edge_df["~to"] = edge_df["to_cols_id"]

        edge_df = edge_df[["~label", "~from", "~to"]]

    edge_df = edge_df.drop_duplicates().reset_index(drop=True)

    # edge_df["~id"] = get_uuid()
    edge_df["~id"] = [get_uuid() for _ in range(len(edge_df.index))]

    for key, value in properties.items():
        if key.find(":") < 0:
            key = key + ":string"
        if value in upd_df.columns:
            edge_df[key] = upd_df[value]
        else:
            edge_df[key] = value

    return edge_df

def generate_embeddings(df):

    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0", client=bedrock_client
    )
    for i in df.index:
        json = df.loc[i].encode('utf-8').to_json(default_handler=str)
        embedding = bedrock_embeddings.embed_query(json)
        df.at[i, "embedding:vector"] = (
            str(embedding).replace(", ", ";").replace("[", "").replace("]", "")
        )

    return df


def generate_gremlin_files(file_location, json_location):
    
    file_name = file_location.split("/")[-1].split(".")[0]
    with open(json_location) as file:
        graph_dict = json.loads(file.read())
    df = pd.read_csv(file_location)
    df = df.fillna("N/A")
    df = whitespace_remover(df)
    df = df.astype(str)
    df = df.add_suffix(':string')
    ic(file_name)

    for node in graph_dict["nodes"]:
        ic(node)
        nodes_df = get_graph_nodes(df, node)

        out_file = node_filename(
            file_name, node.get("label"), node.get("file_name")
        )
        out_path = out_file.split(file_name)[0]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
#         nodes_df = generate_embeddings(nodes_df)
        nodes_df.to_csv(out_file, index=False)

    for edge in graph_dict["edges"]:
        ic(edge)
        from_df, to_df = get_dfs(file_name, edge)
        edges_df = get_connect_edges(df, edge, from_df, to_df)
        out_file = (
            "edges/"
            + (file_name + "_" + edge.get("file_name", edge.get("label")))
            .replace(" ", "_")
            .lower()
        )
        ic(out_file)
        out_path = out_file.split(file_name)[0]
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        edges_df.to_csv(f"{out_file}.csv", index=False)
