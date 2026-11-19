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
f.text(70,45,'やまびこ1号',64)
f.text(74,135,'口笛を聴き、真似をして笛を吹くロボット',32)
f.box(70,240,300,150,'#e2f2f1'); f.text(95,260,'人のお手本',34); f.text(95,320,'口笛・メロディー')
f.line([(370,315),(480,315)])
f.box(480,240,470,255); f.text(510,263,'Arduino UNO Q',39)
f.text(510,330,'Linux：聴き取り・制御・NN補正'); f.text(510,382,'MCU：マイク収録・駆動'); f.text(510,434,'A基板：駆動 ／ B基板：マイク',26)
f.line([(950,315),(1080,315)])
f.box(1080,240,450,255,'#fff0dc'); f.text(1110,265,'スライドホイッスル',34)
f.text(1110,330,'プランジャーで管長を変える'); f.text(1110,385,'ファン＋回転円板弁で発音'); f.text(1110,438,'直動軸に位置センサーなし',27)
f.box(1080,600,450,115,'#e2f2f1'); f.text(1110,615,'I2Sマイク（ICS-43434）',29); f.text(1110,663,'鳴った音程をフィードバック',26)
f.line([(1300,495),(1300,600)])
f.line([(1080,660),(720,660),(720,495)])
f.text(735,555,'自分の音を聴いて、',23,color='#147c83')
f.text(735,590,'ずれを直す',23,color='#147c83')
f.box(70,600,470,115,'#172c43','#172c43'); f.text(95,619,'音を「位置センサー」にする',31,'#ffffff'); f.text(95,669,'聞く → 動かす → 聞き直す',26,'#ffffff')
f.text(74,790,'開発中｜基板製作・部品購入完了 → 組み立て・実機評価へ',30)
f.text(74,845,'構成の模式図です。完成した実機の写真ではありません。',21,'#617286')
f.save('yamabiko-overview')

f=Figure(); f.text(70,45,'まず機体を知り、演奏を仕上げる',52)
f.text(74,125,'デジタルツイン＋決定論制御＋ニューラルネットワークの補正',29)
for x,n,t,sub in [(70,'01','約9秒の較正','音と駆動指令から機体を推定'),(590,'02','デジタルツイン','推定した機体で制御を仕上げる'),(1110,'03','繰り返し演奏','前の演奏を次の補正に使う')]:
    f.box(x,215,420,180); f.text(x+25,232,n,29,'#278c95'); f.text(x+25,280,t,34); f.text(x+25,342,sub,23)
f.line([(490,305),(580,305)]); f.line([(1010,305),(1100,305)])
f.text(75,440,'固定シミュレーション評価：未知の機体32台 × 参照10曲（0.5倍速）',30)
for y,label,value,color in [(525,'決定論＋ツイン',80,'#8294a9'),(615,'NN補正あり・1回目',92,'#278c95'),(705,'NN補正あり・3回目',93,'#e59b3a')]:
    f.text(75,y,label,28); f.box(440,y, value*9,52,color,color); f.text(440+value*9+25,y-2,f'{value} %',36)
f.text(75,807,'命中率：各音の開始から0.3秒以内に、目標の±25 centへ入った割合',25)
f.text(75,850,'資料：2026-09-24時点の計画・結果。数値はシミュレーションであり、実機では未検証。',23,'#617286')
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
