import time
import sys
import numpy as np
import tracemalloc
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
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

def generate_rsa_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    return private_key, public_key


def rsa_encrypt(message, public_key):
    ciphertext = public_key.encrypt(
        message.encode(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext


def rsa_decrypt(ciphertext, private_key):
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext.decode()


def get_memory_size(obj):
    return sys.getsizeof(obj)


def RSA(message, NoOfBlocks):
    if len(message.shape) >= 2:
        ENC_time = []
        DEC_Time = []
        mem_size = []
        Compt_Time = []
        Encrypted_Data = []
        Decrypted_Data = []
        for n in range(len(message)):
            Encry_Data = []
            Decrcry_Data = []
            for i in range(message.shape[1]):
                msg = str(message[n, i])
                tracemalloc.start()
                private_key, public_key = generate_rsa_key_pair()
                ct = time.time()
                rsa_Encrypted = rsa_encrypt(msg, public_key)
                blockChain, blockType = blockchain_with_encryption(NoOfBlocks, msg, rsa_Encrypted)
                ENC_time.append(time.time() - ct)
                mem_size.append(get_memory_size(msg))
                enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
                rsa_decrypted = rsa_decrypt(enc_data, private_key)
                DEC_Time.append(time.time() - ENC_time[n])
                Compt_Time.append(ENC_time[n] + DEC_Time[n])
                Encry_Data.append(enc_data)
                Decrcry_Data.append(rsa_decrypted)
            Encrypted_Data.append(Encry_Data)
            Decrypted_Data.append(Decrcry_Data)
        ENC_time = np.mean(ENC_time, axis=0)
        DEC_Time = np.mean(DEC_Time, axis=0)
        mem_size = np.mean(mem_size, axis=0)
        Compt_Time = np.mean(Compt_Time, axis=0)
    else:
        msg = str(message)
        tracemalloc.start()
        private_key, public_key = generate_rsa_key_pair()
        ct = time.time()
        rsa_Encrypted = rsa_encrypt(msg, public_key)
        blockChain, blockType = blockchain_with_encryption(NoOfBlocks, msg, rsa_Encrypted)
        ENC_time = time.time() - ct
        mem_size = get_memory_size(msg)
        enc_data = blockChain.chain[NoOfBlocks].data['encrypted_data']
        rsa_decrypted = rsa_decrypt(enc_data, private_key)
        DEC_Time = time.time() - ENC_time
        Compt_Time = ENC_time + DEC_Time
        Encrypted_Data = enc_data
        Decrypted_Data = rsa_decrypted
    Encrypted_Data = np.asarray(Encrypted_Data)
    Decrypted_Data = np.asarray(Decrypted_Data)
    return [ENC_time, DEC_Time, mem_size, Compt_Time], Encrypted_Data, Decrypted_Data

