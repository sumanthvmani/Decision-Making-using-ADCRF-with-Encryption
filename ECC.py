from crypto.Cipher import AES
from crypto.Random import get_random_bytes
from crypto.Util.Padding import pad, unpad
import time
import sys
import numpy as np
from BLOCK_CHAIN import Blockchain, Block

# https://github.com/nakov/Practical-Cryptography-for-Developers-Book/blob/master/asymmetric-key-ciphers/ecc-encryption-decryption.md


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


def encrypt_ECC(data, key):
    # Convert int to bytes and then encrypt
    message_bytes = data.to_bytes(8, byteorder='big')  # Assuming 8 bytes for an int
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(message_bytes, AES.block_size))
    return ciphertext


def decrypt_ECC(ciphertext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)
    # Convert decrypted bytes back to int
    decrypted_data = int.from_bytes(decrypted_bytes, byteorder='big')
    return decrypted_data


def get_memory_size(obj):
    return sys.getsizeof(obj)


def ECC(Data, NoOfBlocks):
    if len(Data.shape) >= 2:
        ENCRY_Time = []
        DECRY_Time = []
        MEMO_Size = []
        COMP_Time = []
        Encrypted_Data = []
        Decrypted_Data = []
        for n in range(len(Data)):
            Encry_Data = []
            Decrcry_Data = []
            for i in range(Data.shape[1]):
                Plain = int(Data[n, i])
                key = get_random_bytes(16)  # 16-bit key for ECC
                enc_stat_time = time.time()
                encrypted_data = encrypt_ECC(Plain, key)
                blockChain, blockType = blockchain_with_encryption(NoOfBlocks, Plain, encrypted_data)
                ENC_time = time.time() - enc_stat_time
                dec_stat_time = time.time()
                enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
                decrypted_data = decrypt_ECC(enc_data, key)
                DEC_Time = time.time() - dec_stat_time
                mem_size = get_memory_size(Plain)
                Compt_Time = ENC_time + DEC_Time
                ENCRY_Time.append(ENC_time)
                DECRY_Time.append(DEC_Time)
                MEMO_Size.append(mem_size)
                COMP_Time.append(Compt_Time)
                Encry_Data.append(enc_data)
                Decrcry_Data.append(decrypted_data)
            Encrypted_Data.append(Encry_Data)
            Decrypted_Data.append(Decrcry_Data)
        ENC_time = np.mean(ENCRY_Time, axis=0)
        DEC_Time = np.mean(DECRY_Time, axis=0)
        mem_size = np.mean(MEMO_Size, axis=0)
        Compt_Time = np.mean(COMP_Time, axis=0)
    else:
        Plain = int(Data)
        key = get_random_bytes(16)  # 16-bit key for ECC
        enc_stat_time = time.time()
        encrypted_data = encrypt_ECC(Plain, key)
        blockChain, blockType = blockchain_with_encryption(NoOfBlocks, Plain, encrypted_data)
        ENC_time = time.time() - enc_stat_time
        dec_stat_time = time.time()
        enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
        decrypted_data = decrypt_ECC(enc_data, key)
        DEC_Time = time.time() - dec_stat_time
        mem_size = get_memory_size(Plain)
        Compt_Time = ENC_time + DEC_Time
        Encrypted_Data = enc_data
        Decrypted_Data = decrypted_data
    Encrypted_Data = np.asarray(Encrypted_Data)
    Decrypted_Data = np.asarray(Decrypted_Data)
    return [ENC_time, DEC_Time, mem_size, Compt_Time], Encrypted_Data, Decrypted_Data



