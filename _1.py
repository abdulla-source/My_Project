import numpy as np


def encode(encode_map, data):
  return np.array([encode_map.get(c, 0) for c in data]) # Changed None to 0


def embedding(input, mat):
  return mat[input]


def layer(X, W, b):
  return X @ W + b, (W, b)

def GELU(Z):
  return 0.5 * Z * (1 + np.tanh(np.sqrt(2/np.pi) * (Z + 0.044715 * (Z**3))))


def softmax(Z):
  # Add a small value to prevent division by zero and handle large exponents
  exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
  return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

def GELU_deriv(Z):
  tanh_term = np.tanh(np.sqrt(2/np.pi)*(Z + 0.044715 * (Z**3)))
  return 0.5 * (1 + tanh_term) + 0.5 * Z * (1 - tanh_term**2) * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * (Z**2))


def loss(pred, target):
  
    # Convert target to one-hot
    y_true = np.zeros_like(pred)
    y_true[np.arange(len(target)), target] = 1

    # Cross entropy loss (avoid log(0) by adding small epsilon)
    epsilon = 1e-9
    loss_value = -np.sum(y_true * np.log(pred + epsilon)) / len(target)

    return loss_value 



# Modified backprop function to accept necessary variables and return updated weights/biases
def backprop(pred, targets, H1, W_out, out_b, lr, Z1, W1, b1,  X_embedded): # Added X_embedded argument
  y_true = np.zeros_like(pred)
  y_true[np.arange(len(targets)), targets] = 1

  dZ_out = (pred - y_true)/len(targets)
  dW_out = np.dot(H1.T, dZ_out)
  db_out = np.sum(dZ_out, axis = 0, keepdims=True)

  dH1 = np.dot(dZ_out, W_out.T)
  dZ1 = dH1 * GELU_deriv(Z1)
  dW1 = np.dot(X_embedded.T, dZ1) # Corrected calculation for dW1
  db1 = np.sum(dZ1, axis = 0, keepdims = True)

  # Update weights and biases
  W1 -= lr * dW1
  b1 -= lr * db1
  W_out -= lr * dW_out
  out_b -= lr * db_out

  return W1, b1, W_out, out_b





import matplotlib.pyplot as plt

data = """
Ertalab tong yorishdi. Ko‘chada hali ham sokinlik bor edi. Daraxt barglari orasidan quyoshning ilk nurlari tushar, qushlarning mayin chinqiriqlari eshitilardi. Men bugun erta uyg‘ondim, chunki oldimda katta vazifa bor: o‘zim yaratgan sun’iy intellekt modelini yana bir bor o‘qitib ko‘rishim kerak. Kecha loss qiymati 3.85 dan pastga tushmadi. Balki matn yetarli bo‘lmagandir, yoki embedding to‘g‘ri ishlamayotgandir, balki og‘irliklar noto‘g‘ri initsializatsiya qilingandir. Har holda taslim bo‘lish fikrim yo‘q.

Kompyuter yonida choy ichib o‘tirdim va kodga yana bir bor nazar tashladim. Embedding matriksa 256x64, ya’ni har bir byte belgisi 64 o‘lchamli vektor ko‘rinishiga o‘tayapti. Bu yaxshi, chunki model belgi darajasida ishlaydi. Keyin bu vektorlar yashirin qatlamga o‘tadi, u yerda 64x128 og‘irlik matritsasi bilan ko‘payadi, ustiga bias qo‘shiladi va GELU aktivatsiyasidan o‘tadi. GELU – bu ReLU ga o‘xshash, lekin silliqroq funksiya, qiymatlarni normal taqsimot asosida yumshatadi. Shuning uchun ko‘pchilik transformer modellarida shu ishlatiladi.

Men terminalni ochdim, kodni ishga tushirdim. Har bir epoch tugagach, loss ekranga chiqadi. Birinchi epoch: 4.932. Ikkinchisi: 4.621. Keyin sekin-asta 4.2, 4.1, lekin keyin to‘xtab qoldi. Balki dataset juda kichikdir. Shuning uchun men model uchun yangi matn yozishga qaror qildim. Tabiiy, silliq, real odam gapiradigan uslubda bo‘lsin deb niyat qildim.

Men eslayman, bolaligimda yoz kunlari qishloqqa borardik. U yerda havo boshqacha edi, osmon toza, yulduzlar yorqin. Kechasi quyonlar, mushuklar, itlar ovozi eshitilardi. Buvim doimo shunday derdi: “Har bir narsaning o‘z vaqti bor, bolasim. Daraxt ham bir kunda o‘sib qolmaydi, odam ham shunday.” Men hozir shu so‘zlarni eslab, o‘zimga shunday deyman: “Model ham birdaniga ideal bo‘lmaydi. Har bir qadam kichik yutuq.”

Do‘stim Dilshod menga yozdi:
— Nima bo‘lyapti, model ishlayaptimi?
— Hali uncha emas, — dedim men. — Grammatikani tushunmayapti, ba’zi so‘zlarni qaytaraveradi.
— Dataset yetmaydi, — dedi u. — Senga dialog, hikoya, kundalik, hatto xato yozilgan gaplar ham kerak.
— Bilaman, shuning uchun o‘zim yozyapman. Kamida 10 ming token yig‘moqchiman.

Shu payt deraza ortidan mayin shamol esdi. Men sekin o‘rnimdan turdim dan chiroyli osmonga qaradim. Osmonda bulutlar suzib yurardi. Shunday lahzalarda inson o‘zini tinch his qiladi. Ammo mening miyamda faqat bitta narsa: embedding, loss, gradientlar, learning rate, softmax, cross entropy.

Softmax funksiyasi chiqishdagi 128 o‘lchamli vektor orqali 255 ta ehtimollik qiymati yaratadi. Men aslida bu yerda 255 emas, 256 bo‘lishi kerakligini tushunib yetdim. Chunki byte qiymatlari 0 dan 255 gacha. Bitta token yo‘qolgan bo‘lsa, model hech qachon to‘liq o‘rganmaydi. Shuning uchun bugun kodni to'g'rilayman.

Kecha tushda bir g'alati tush ko'rdim. Go'yo kompyuter ekranidan matnlar chiqib, havoda suzib yurar, harflar bir-biriga ulanib, so'zga aylanar, so'zlar esa gapga. Ularning ichida bu so'zlar yozilgan edi: "Davom et. To'xtama. Sen yaratgan narsa bir kun boshqalarga ham foyda beradi."

Bu tush meni yanada ilhomlantirdi. Endi men yana klaviaturaga qaytaman. Yana matn yozaman. Chunki model so'zlarni tushunishi uchun avvalo so'zlar bo'lishi kerak.

Bugun men nafaqat kod yozdim, balki o'zimni ham tushundim. Sun'iy intellekt — bu faqat matematik formulalar emas. Bu — sabr, kreativlik, mantiq va inson qalbining chidamliligi.
"""
encode_map = {chr(i) : i for i in range(256)}

encoded_data = encode(encode_map, data)
inputs = encoded_data[:-1]
targets = encoded_data[1:]


mat = np.random.randn(256, 64)
W1 = np.random.randn(64, 128) * 0.01
W_out = np.random.randn(128, 256)
b1 = np.zeros((1, 128))
b_out = np.zeros((1, 256))

epochs = 150
lr = 0.001
loss_history = []

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(inputs)):
        X = np.array([inputs[i]])
        target = np.array([targets[i]])

        # Forward pass
        X_embedded = embedding(X, mat)
        Z1, _ = layer(X_embedded, W1, b1)
        H1 = GELU(Z1)
        Z_out, _ = layer(H1, W_out, b_out)
        pred = softmax(Z_out)

        total_loss += loss(pred, target)

        # Backpropagation
        W1, b1, W_out, b_out = backprop(pred, target, H1, W_out, b_out, lr, Z1, W1, b1, X_embedded)

    avg_loss = total_loss / len(inputs)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# Plot
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()





def generate_text(start_char, mat, W1, b1, W_out, b_out, encode_map, num_chars=100):
    decode_map = {v: k for k, v in encode_map.items()}
    encoded_input = encode(encode_map, start_char)
    generated_text = start_char

    for _ in range(num_chars):
        X = np.array([encoded_input[-1]]) # Use the last character as input

        X_embedded = embedding(X, mat)
        Z1, _ = layer(X_embedded, W1, b1)
        H1 = GELU(Z1)
        Z_out, _ = layer(H1, W_out, b_out)
        pred = softmax(Z_out)

        # Sample the next character based on probabilities
        next_char_encoded = np.random.choice(len(encode_map), p=pred.flatten())
        next_char = decode_map.get(next_char_encoded, '')

        generated_text += next_char
        encoded_input = np.append(encoded_input, next_char_encoded) # Append the new character to the input sequence

    return generated_text

# Generate text starting with "Ertalab"
generated = generate_text("Ertalab", mat, W1, b1, W_out, b_out, encode_map, num_chars=200)
print(generated)
