import hashlib
import json
import time
import random
import numpy as np


def to_json_safe(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, dict):
        return {k: to_json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_json_safe(v) for v in data]
    return data


def calculate_hash(block):
    safe_block = to_json_safe(block)
    block_string = json.dumps(safe_block, sort_keys=True).encode()
    return hashlib.sha256(block_string).hexdigest()


def create_block(index, data, previous_hash):
    block = {
        "index": index,
        "timestamp": time.time(),
        "data": to_json_safe(data),
        "previous_hash": previous_hash
    }
    block["hash"] = calculate_hash(block)
    return block


def validate_block(block, faulty=False):
    if faulty:
        return random.choice([True, False])

    recalculated_hash = calculate_hash({
        "index": block["index"],
        "timestamp": block["timestamp"],
        "data": block["data"],
        "previous_hash": block["previous_hash"]
    })
    return block["hash"] == recalculated_hash


def pbft_consensus(block, nodes):
    votes = 0
    f = (len(nodes) - 1) // 3

    for node in nodes:
        if validate_block(block, node["faulty"]):
            votes += 1

    return votes >= (2 * f + 1)


def pbft_blockchain(Data):
    nodes = [
        {"id": 1, "faulty": False},
        {"id": 2, "faulty": False},
        {"id": 3, "faulty": False},
        {"id": 4, "faulty": True}
    ]
    blockchain = []
    # Genesis Block
    blockchain.append(create_block(0, "Genesis Block", "0"))
    # Convert once for safe iteration
    safe_data = to_json_safe(Data)
    # Add Data Blocks
    for sample in safe_data:
        prev_block = blockchain[-1]
        new_block = create_block(
            index=len(blockchain),
            data=sample,
            previous_hash=prev_block["hash"]
        )

        if pbft_consensus(new_block, nodes):
            blockchain.append(new_block)
    # Extract secured data (skip genesis)
    secured_list = [block["data"] for block in blockchain[1:]]
    SecuredData = np.array(secured_list, dtype=np.float64)
    return SecuredData


def BlockChain(Data, Target):
    secured_data = pbft_blockchain(Data.astype(np.float64))
    secured_Target = pbft_blockchain(Target.astype(np.float64))
    return secured_data, secured_Target.astype('int')
