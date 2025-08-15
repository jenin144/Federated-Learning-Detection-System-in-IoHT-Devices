run : python main.py 

**********************

WARNING: The input does not fit in a single ciphertext, and some operations will be disabled.
The following operations are disabled in this setup: matmul, matmul_plain, enc_matmul_plain, conv2d_im2col.
->

🔵 This is just a benign warning from TenSEAL indicating that the encrypted vector is too large to fit in a single ciphertext, disabling some specific operations we’re not using anyway (like encrypted matrix multiplication). It won’t affect our current federated learning workflow. 🔵

