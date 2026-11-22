from pathlib import Path
import html, math, sys
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
FONT = 'C:/Windows/Fonts/meiryo.ttc'
class Figure:
    def __init__(self):
        self.im = Image.new('RGB', (1600,900), '#f5f7fb')
        self.d = ImageDraw.Draw(self.im)
        self.svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900"><rect width="1600" height="900" fill="#f5f7fb"/>']
    def box(self,x,y,w,h,fill='#ffffff',stroke='#d2dce8'):
        self.d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=fill,outline=stroke,width=3)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    def text(self,x,y,s,size=28,color='#172c43'):
        self.d.text((x,y),s,font=ImageFont.truetype(FONT,size),fill=color)
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="Meiryo,sans-serif" font-size="{size}" fill="{color}">{html.escape(s)}</text>')
    def line(self,points,color='#278c95',width=6,arrow=True):
        self.d.line(points,fill=color,width=width)
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}"/>')
        if arrow:
            x,y=points[-1]; a=math.atan2(y-points[-2][1],x-points[-2][0]); p=[(x,y),(x-17*math.cos(a-.5),y-17*math.sin(a-.5)),(x-17*math.cos(a+.5),y-17*math.sin(a+.5))]
            self.d.polygon(p,fill=color)
            self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in p)}" fill="{color}"/>')
    def save(self,name):
        (OUT/f'{name}.svg').write_text(''.join(self.svg)+'</svg>',encoding='utf-8')
        self.im.save(OUT/f'{name}.png')

f=Figure()
f.text(70,40,'やまびこ1号',60)
f.text(74,122,'口笛を聴き、AI がスライドホイッスルでまねて吹くロボット',30)
f.box(70,215,280,140,'#e2f2f1'); f.text(95,235,'人の口笛',32); f.text(95,292,'メロディーのお手本',24)
f.box(70,420,280,170,'#e2f2f1'); f.text(95,440,'MEMSマイク',30); f.text(95,492,'I2S（ICS-43434）',24); f.text(95,532,'口笛と笛の音程を聴く',22)
f.line([(210,355),(210,420)])
f.box(450,215,560,470); f.text(480,232,'Arduino UNO Q',38)
f.box(475,300,510,175,'#eef4ff','#9db6e8'); f.text(500,314,'Linux側',28); f.text(500,358,'耳（YIN）：10 ms ごとに音程',23); f.text(500,396,'NN（GRU）：駆動信号を直接出力',23); f.text(500,434,'口笛コピー・ウェブ画面',23)
f.box(475,495,510,165,'#eef4ff','#9db6e8'); f.text(500,509,'マイコン側（STM32）',28); f.text(500,553,'I2S取り込み・PWM・サーボ信号',23); f.text(500,591,'100 Hzの指令・安全停止',23)
f.text(480,630,'A基板：駆動（TB6643）／B基板：マイク',21,'#617286')
f.line([(350,505),(450,505)])
f.box(1110,180,430,150,'#fff0dc'); f.text(1135,195,'DCリニアアクチュエータ',28); f.text(1135,245,'エンコーダなし',22); f.text(1135,280,'ピストン → 管体の長さ＝音程',22)
f.box(1110,350,430,110,'#fff0dc'); f.text(1135,365,'遠心ファン',28); f.text(1135,410,'吹き口へ空気を送る',22)
f.box(1110,480,430,130,'#fff0dc'); f.text(1135,495,'SG90 → 自作の切換え弁',28); f.text(1135,545,'タンギング（空気の切換え）',22)
f.box(1110,630,430,100,'#fff0dc'); f.text(1135,648,'スライドホイッスル',30); f.text(1135,690,'管体・ピストン・吹き口',21)
for y in (255,405,545):
    f.line([(1010,y),(1110,y)])
f.line([(1110,700),(380,700),(380,560)],color='#e2672a')
f.text(420,715,'自分の音を聴いて直す（音程を位置センサーの代わりにする）',24,color='#c2410c')
f.box(70,770,1470,90,'#172c43','#172c43'); f.text(100,790,'Step1：NN が DC リニアアクチュエータの駆動信号を出し、口笛をまねて吹く（実機で動作）',28,'#ffffff')
f.save('yamabiko-overview')

f=Figure(); f.text(70,40,'デジタルツインで学び、NN が駆動信号を出す',48)
f.text(74,112,'エンコーダなしの DC リニアアクチュエータを、聞こえた音程だけで制御する',27)
for x,n,t,sub in [(70,'01','実機で較正','全行程 約14秒、音と指令を記録'),(455,'02','デジタルツイン','静止摩擦・管の長さ・速さ'),(840,'03','シミュレーターで学習','音程誤差＋駆動信号の滑らかさ'),(1225,'04','UNO Q で推論','NumPy、1ステップ約1 ms')]:
    f.box(x,180,330,170); f.text(x+22,196,n,27,'#278c95'); f.text(x+22,238,t,29); f.text(x+22,296,sub,20)
for x in (400,785,1170):
    f.line([(x,265),(x+55,265)])
f.text(75,395,'実機の結果（2026-11-22、ドレミファミレド）',30)
f.text(75,455,'1曲で駆動信号の向きが反転した回数',25)
for y,label,value,color in [(500,'従来の制御（PD＋位置の推定器）',300,'#8294a9'),(570,'NN の制御（Step1）',9,'#278c95')]:
    f.text(75,y,label,24); f.box(560,y,max(value*2.6,10),46,color,color); f.text(560+max(value*2.6,10)+20,y,f'{value} 回',30)
f.box(75,650,690,130); f.text(100,665,'隣り合う音の音程の誤差',25); f.text(100,710,'13〜16 cent',40,'#278c95')
f.box(830,650,690,130); f.text(855,665,'音程が取れた割合',25); f.text(855,710,'99 %',40,'#278c95')
f.text(75,815,'100 cent ＝ 半音。従来の制御は静止摩擦を越えるため、細かく上下を繰り返していた。',22,'#617286')
f.save('yamabiko-control-results')

sys.path.insert(0,str(OUT.parents[1]/'hardware'/'gen'))
from design import A_PARTS,B_PARTS
rows=['# 基板BoM（設計データから集計）\n','数量は基板各1枚あたり。予備回路のDNP部品は別記。コネクターの相手側ハウジング・圧着端子、配線、ICソケット等は別途必要。購入価格の記録がないため金額は記載しない。\n']
for name,parts in [('A基板',A_PARTS),('B基板',B_PARTS)]:
    rows += [f'\n## {name}\n','|部品・値|数量|リファレンス|秋月商品番号|実装|','|---|---:|---|---|---|']
    groups=defaultdict(list)
    for p in parts:
        value=p.value
        if p.sym == 'Device:R': value='抵抗 '+value.replace('k',' kΩ')+(' Ω' if 'k' not in value else '')
        if p.sym in ('Device:C','Device:C_Polarized'): value=('電解コンデンサー ' if p.sym.endswith('Polarized') else 'コンデンサー ')+value.replace('u',' µF')
        if p.ref in ('J101','J102','J103','J104'): value='ピンヘッダー '+{'J101':'1×8','J102':'1×10','J103':'1×8','J104':'1×6'}[p.ref]
        elif p.ref in ('J14','J15'): value='裏面実装ピンソケット 2×20'
        elif p.device and p.ref.startswith('J'): value='XH '+str(len(p.pins))+'極（'+p.device+'）'
        elif p.ref=='SW1': value='タクトスイッチ（演奏開始）'
        if p.fp: groups[(value,p.akizuki,p.dnp)].append(p.ref)
    for (value,code,dnp),refs in groups.items():
        rows.append(f'|{value}|{len(refs)}|{", ".join(refs)}|{code}|{"DNP（予備）" if dnp else "実装対象"}|')
(OUT/'board-bom.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
print('Created two SVG/PNG figures and board-bom.md')
