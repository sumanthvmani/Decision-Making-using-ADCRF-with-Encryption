import numpy as np
from crypto.Cipher import DES as des
from crypto.Util.Padding import pad, unpad
import time
import tracemalloc
import sys
from BLOCK_CHAIN import Blockchain, Block


def blockchain_with_encryption(no_of_blocks, data, encrypted_ElGamal):
    blockchain = Blockchain()
    blockType = []
    for i in range(no_of_blocks):
        # Add data to the blockchain
        block_data = {
            "original_data": data,
            "encrypted_data": encrypted_ElGamal,
        }
        new_block = Block(i + 1, block_data, blockchain.chain[-1].hash)
        blockchain.add_block(new_block)
        blockType.append(1 if i % 5 == 0 else 0)
    return blockchain, blockType

def encrypt_des(plaintext, key):
    cipher = des.new(key, des.MODE_ECB)
    padded_plaintext = pad(plaintext.encode(), des.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext

def decrypt_des(ciphertext, key):
    cipher = des.new(key, des.MODE_ECB)
    decrypted_plaintext = cipher.decrypt(ciphertext)
    unpadded_plaintext = unpad(decrypted_plaintext, des.block_size)
    return unpadded_plaintext.decode()


def get_memory_size(obj):
    return sys.getsizeof(obj)


def DES(Data, NoOfBlocks):
    if len(Data.shape) >= 2:
        ENC_time = []
        DEC_Time = []
        mem_size = []
        Compt_Time = []
        Encrypted_Data = []
        Decrypted_Data = []
        for n in range(len(Data)):
            Encry_Data = []
            Decrcry_Data = []
            for i in range(Data.shape[1]):
                MSG = Data[n, i]
                MSG = str(MSG)
                key = b"01234567"  # 8-byte key
                tracemalloc.start()
                start_enc_t = time.time()
                ciphertext = encrypt_des(MSG, key)
                blockChain, blockType = blockchain_with_encryption(NoOfBlocks, MSG, ciphertext)
                ENC_time.append(time.time() - start_enc_t)
                start_dec_t = time.time()
                enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
                decrypted_plaintext = decrypt_des(enc_data, key)
                mem_size.append(get_memory_size(MSG))
                DEC_Time.append(time.time() - start_dec_t)
                Compt_Time.append(ENC_time[n] + DEC_Time[n])
                Encry_Data.append(enc_data)
                Decrcry_Data.append(decrypted_plaintext)
            Encrypted_Data.append(Encry_Data)
            Decrypted_Data.append(Decrcry_Data)
        ENC_time = np.mean(ENC_time, axis=0)
        DEC_Time = np.mean(DEC_Time, axis=0)
        mem_size = np.mean(mem_size, axis=0)
        Compt_Time = np.mean(Compt_Time, axis=0)
    else:
        Data = str(Data)
        key = b"01234567"  # 8-byte key
        tracemalloc.start()
        start_enc_t = time.time()
        ciphertext = encrypt_des(Data, key)
        blockChain, blockType = blockchain_with_encryption(NoOfBlocks, Data, ciphertext)
        ENC_time = time.time() - start_enc_t
        start_dec_t = time.time()
        enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
        decrypted_plaintext = decrypt_des(enc_data, key)
        mem_size = get_memory_size(Data)
        DEC_Time = time.time() - start_dec_t
        Compt_Time = ENC_time + DEC_Time
        Encrypted_Data = enc_data
        Decrypted_Data = decrypted_plaintext
    Encrypted_Data = np.asarray(Encrypted_Data)
    Decrypted_Data = np.asarray(Decrypted_Data)
    return [ENC_time, DEC_Time, mem_size, Compt_Time], Encrypted_Data, Decrypted_Data

