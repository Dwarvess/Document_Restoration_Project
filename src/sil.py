import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Figür oluşturma
fig, ax = plt.subplots(figsize=(8, 8))

# Çizim oranları için temsili parametreler
W = 1.5  # Yama genişliği
L = 1.0  # Yama uzunluğu
dx = 3.0 # X eksenindeki elemanlar arası mesafe
dy = 3.0 # Y eksenindeki elemanlar arası mesafe

# 4 adet yamanın merkez koordinatları (2x2 Matris)
centers = [
    (-dx/2, dy/2),   # Sol Üst
    (dx/2, dy/2),    # Sağ Üst
    (-dx/2, -dy/2),  # Sol Alt
    (dx/2, -dy/2)    # Sağ Alt
]

# Yamaları (Dikdörtgenleri) çiz
for cx, cy in centers:
    rect = patches.Rectangle((cx - W/2, cy - L/2), W, L, 
                             linewidth=2, edgecolor='black', facecolor='skyblue')
    ax.add_patch(rect)

# Besleme Ağı (Corporate Feed Network)
# Üstteki iki anteni birleştiren hat
ax.plot([-dx/2, dx/2], [dy/2 - L/2, dy/2 - L/2], color='black', linewidth=2)
# Alttaki iki anteni birleştiren hat
ax.plot([-dx/2, dx/2], [-dy/2 + L/2, -dy/2 + L/2], color='black', linewidth=2)
# Üst ve alt grupları birleştiren dikey hat
ax.plot( [dy/2 - L/2, -dy/2 + L/2], color='black', linewidth=2)
# Ana giriş (Besleme) hattı
ax.plot( [-dy/2 + L/2, -dy/2 + L/2 - 1.5], color='black', linewidth=3)

# Tasarım Parametrelerinin Şema Üzerinde Gösterimi
# L (Uzunluk) ve W (Genişlik)
ax.annotate('', xy=(-dx/2 - W/2, dy/2 - L/2), xytext=(-dx/2 - W/2, dy/2 + L/2), arrowprops=dict(arrowstyle='<->'))
ax.text(-dx/2 - W/2 - 0.4, dy/2, '$L$', fontsize=14, va='center')

ax.annotate('', xy=(-dx/2 - W/2, dy/2 + L/2 + 0.2), xytext=(-dx/2 + W/2, dy/2 + L/2 + 0.2), arrowprops=dict(arrowstyle='<->'))
ax.text(-dx/2, dy/2 + L/2 + 0.4, '$W$', fontsize=14, ha='center')

# dx ve dy (Merkezler Arası Mesafeler)
ax.annotate('', xy=(-dx/2, dy/2 + L/2 + 1.0), xytext=(dx/2, dy/2 + L/2 + 1.0), arrowprops=dict(arrowstyle='<->', color='red'))
ax.text(0, dy/2 + L/2 + 1.2, '$d_x$', fontsize=14, ha='center', color='red')

ax.annotate('', xy=(dx/2 + W/2 + 1.0, dy/2), xytext=(dx/2 + W/2 + 1.0, -dy/2), arrowprops=dict(arrowstyle='<->', color='red'))
ax.text(dx/2 + W/2 + 1.2, 0, '$d_y$', fontsize=14, va='center', color='red')

# Giriş Empedansı ve Hat Kalınlığı
ax.text(0.2, -dy/2 + L/2 - 1.2, '$Z_{in}$ (Giriş Empedansı)', fontsize=12)
ax.text(0.2, 0, '$W_f$ (Besleme Hattı Kalınlığı)', fontsize=12)

# Eksenleri gizle ve başlık ekle
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.axis('off')
plt.title("2x2 Dikdörtgensel Dizi Anten Şematik Gösterimi", fontsize=16)
plt.tight_layout()

# Çizimi göster
plt.show()