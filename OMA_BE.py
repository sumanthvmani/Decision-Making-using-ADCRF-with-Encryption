import numpy as np
from crypto.Cipher import AES
from crypto.Util.Padding import pad, unpad
import time
import tracemalloc
import sys
from BLOCK_CHAIN import Blockchain, Block


# Blockchain with encrypted data storage
def blockchain_with_encryption(no_of_blocks, data, encrypted_data):
    blockchain = Blockchain()
    blockType = []
    for i in range(no_of_blocks):
        # Add data to the blockchain
        block_data = {
            "original_data": data,
            "encrypted_data": encrypted_data,
        }
        new_block = Block(i + 1, block_data, blockchain.chain[-1].hash if blockchain.chain else None)
        blockchain.add_block(new_block)
        blockType.append(1 if i % 5 == 0 else 0)
    return blockchain, blockType


# Generate a 16-bit binary key (only 0s and 1s)
def generate_16bit_key():
    # Generate a 16-bit key, randomly populated with 0s and 1s
    key = np.random.randint(0, 2, 16)  # 16 random binary digits (0 or 1)
    # Convert the key to bytes (needed for AES encryption)
    key_bytes = bytes(key.astype(np.uint8))  # Convert the binary array to a bytes object
    return key_bytes


# Encryption with AES (using 16-bit key)
def encrypt_aes(plaintext, key):
    cipher = AES.new(key, AES.MODE_CBC)
    padded_plaintext = pad(plaintext.encode(), AES.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)
    return cipher.iv, ciphertext  # Return IV and ciphertext for AES


def decrypt_aes(iv, ciphertext, key):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return decrypted_plaintext.decode()


# Helper to get memory size of objects
def get_memory_size(obj):
    return sys.getsizeof(obj)


def OMA_BE(Data, NoOfBlocks, sol=None):
    if sol is None:
        sol = generate_16bit_key()
    Key = sol
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
                MSG = str(Data[n, i])

                final_key = Key
                tracemalloc.start()
                start_enc_t = time.time()
                iv, ciphertext = encrypt_aes(MSG, final_key)
                blockChain, blockType = blockchain_with_encryption(NoOfBlocks, MSG, ciphertext)
                ENC_time.append(time.time() - start_enc_t)

                start_dec_t = time.time()
                enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
                decrypted_plaintext = decrypt_aes(iv, enc_data, final_key)

                mem_size.append(get_memory_size(MSG))
                DEC_Time.append(time.time() - start_dec_t)
                Compt_Time.append(ENC_time[n] + DEC_Time[n])

                Encry_Data.append(enc_data)
                Decrcry_Data.append(decrypted_plaintext)

            Encrypted_Data.append(Encry_Data)
            Decrypted_Data.append(Decrcry_Data)

        # Averaging the times
        ENC_time = np.mean(ENC_time, axis=0)
        DEC_Time = np.mean(DEC_Time, axis=0)
        mem_size = np.mean(mem_size, axis=0)
        Compt_Time = np.mean(Compt_Time, axis=0)

    else:
        MSG = str(Data)
        final_key = generate_16bit_key()
        tracemalloc.start()
        start_enc_t = time.time()
        iv, ciphertext = encrypt_aes(MSG, final_key)
        blockChain, blockType = blockchain_with_encryption(NoOfBlocks, MSG, ciphertext)
        ENC_time = time.time() - start_enc_t

        start_dec_t = time.time()
        enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
        decrypted_plaintext = decrypt_aes(iv, enc_data, final_key)

        mem_size = get_memory_size(MSG)
        DEC_Time = time.time() - start_dec_t
        Compt_Time = ENC_time + DEC_Time

        Encrypted_Data = enc_data
        Decrypted_Data = decrypted_plaintext

    Encrypted_Data = np.asarray(Encrypted_Data)
    Decrypted_Data = np.asarray(Decrypted_Data)
    return [ENC_time, DEC_Time, mem_size, Compt_Time], Encrypted_Data, Decrypted_Data
