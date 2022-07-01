from math import sqrt

def draw(node,member,free,A,name): # !!! 入力にfreeを追加した（freeに含まれない番号の節点はピン支持）!!!
  # 解析モデルを図示
  import matplotlib.pyplot as plt # 図を描画するためのライブラリを使用することを宣言
  for i in range(member.shape[0]): # 全ての部材について繰り返し
    if A[i] < 1e-5:
      pass
    else:
      plt.plot(node[member[i,:],0],node[member[i,:],1],color='gray',linewidth=1.5*sqrt(A[i])) # 線分を描画
  for i in range(node.shape[0]): # 全ての節点について繰り返し
    if i in free:
      plt.plot(node[i,0],node[i,1],color='black',marker='o',markersize=8) # 節点を黒丸で描画
      plt.plot(node[i,0],node[i,1],color='white',marker='o',markersize=4) # 小さな白丸を黒丸の上から描画
    else:
      plt.plot(node[i,0],node[i,1],color='black',marker='>',markersize=15) 
  plt.xlabel("$x$ [m]") # 図にx軸ラベルを記載
  plt.ylabel("$y$ [m]") # 図にy軸ラベルを記載
  plt.gca().set_aspect('equal', adjustable='box') # x軸・y軸の表示を同じスケールに
  plt.savefig(f"results/{name}.png")
  # plt.show() # 図を表示
  plt.clf()
