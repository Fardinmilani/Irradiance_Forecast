from pathlib import Path
from io import StringIO, BytesIO

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib import ticker as mticker
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

Features = ['time_of_day', 'Temperature', 'Humidity', 'daylength', 'WindDirection(Degrees)', 'month', 'Pressure', 'Speed']
P_Features = ['Pressure', 'time_of_day', 'Temperature', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'month', 'daylength']
P_Lables = ['Pressure (InHg)', 'Time of day (h)', 'Temperature (°F)', 'Humidity (%)', 'Wind direction (Degrees)', 'Speed (mph)', 'Month of year', 'Day length (min)']

RefView = dict(elev=23.20, azim=-148.53)
Ref_BoxAspect = (1.0, 1.063, 0.811)
REFERENCE_LABEL_POSITIONS = [
    ((.18563,.05100),(.72201,.08333),(.02519,.47015)),
    ((.21175,.04975),(.73228,.08458),(.02612,.47264)),
    ((.21362,.04975),(.75280,.08458),(.02705,.47264)),
    ((.21503,.04986),(.73800,.08449),(.02818,.49446)),
    ((.21175,.04975),(.64832,.03358),(.02612,.47015)),
    ((.21455,.04975),(.73228,.08458),(.02705,.47015)),
    ((.21127,.05014),(.74547,.08401),(.02515,.47019)),
    ((.20989,.04975),(.76399,.08333),(.02425,.47512)),
]

PALETTE = """
0,0,136
0,0,139
0,0,143
0,0,147
0,0,150
0,0,151
0,0,155
0,0,158
0,0,160
0,0,163
0,0,167
0,0,171
0,0,174
0,0,177
0,0,179
0,0,183
0,0,187
0,0,191
0,0,194
0,0,195
0,0,199
0,0,201
0,0,204
0,0,207
0,0,211
0,0,215
0,0,218
0,0,220
0,0,223
0,0,227
0,0,231
0,0,234
0,0,237
0,0,239
0,0,242
0,0,245
0,0,247
0,0,251
0,0,255
0,4,255
0,6,255
0,9,255
0,12,255
0,16,255
0,20,255
0,23,255
0,25,255
0,28,255
0,30,255
0,33,255
0,36,255
0,40,255
0,44,255
0,47,255
0,50,255
0,52,255
0,56,255
0,60,255
0,64,255
0,66,255
0,69,255
0,72,255
0,76,255
0,76,255
0,80,255
0,84,255
0,88,255
0,90,255
0,93,255
0,96,255
0,100,255
0,104,255
0,107,255
0,110,255
0,112,255
0,116,255
0,120,255
0,120,255
0,124,255
0,128,255
0,130,255
0,133,255
0,135,255
0,139,255
0,143,255
0,147,255
0,149,255
0,152,255
0,155,255
0,159,255
0,163,255
0,163,255
0,167,255
0,171,255
0,174,255
0,176,255
0,179,255
0,183,255
0,187,255
0,190,255
0,193,255
0,195,255
0,199,255
0,203,255
0,207,255
0,207,255
0,211,255
0,214,255
0,217,255
0,220,255
0,223,255
0,227,255
0,231,255
0,234,255
0,236,255
0,239,255
0,243,255
0,247,255
0,250,255
0,251,255
0,255,255
3,255,251
5,255,249
8,255,246
12,255,243
16,255,239
19,255,235
22,255,232
25,255,229
28,255,227
32,255,223
36,255,219
39,255,215
40,255,215
44,255,211
46,255,208
49,255,205
52,255,203
56,255,199
60,255,195
63,255,191
65,255,189
68,255,186
72,255,183
76,255,179
80,255,175
82,255,172
84,255,171
87,255,167
90,255,164
92,255,162
96,255,159
100,255,155
104,255,151
106,255,148
109,255,145
112,255,143
116,255,139
120,255,135
123,255,131
125,255,129
128,255,128
130,255,125
132,255,122
135,255,119
139,255,116
143,255,112
146,255,108
149,255,105
151,255,103
155,255,100
159,255,96
163,255,92
165,255,89
168,255,86
171,255,84
175,255,80
175,255,79
179,255,76
183,255,72
187,255,68
189,255,65
192,255,62
195,255,59
199,255,56
203,255,52
206,255,48
209,255,45
211,255,43
215,255,40
219,255,36
219,255,35
223,255,32
227,255,28
230,255,24
233,255,21
235,255,19
239,255,16
243,255,12
247,255,8
250,255,5
252,255,2
255,254,0
255,251,0
255,247,0
255,247,0
255,243,0
255,239,0
255,235,0
255,233,0
255,230,0
255,227,0
255,223,0
255,219,0
255,216,0
255,214,0
255,211,0
255,207,0
255,203,0
255,203,0
255,199,0
255,195,0
255,192,0
255,189,0
255,187,0
255,183,0
255,179,0
255,175,0
255,173,0
255,170,0
255,167,0
255,163,0
255,159,0
255,159,0
255,155,0
255,151,0
255,149,0
255,146,0
255,143,0
255,139,0
255,135,0
255,132,0
255,130,0
255,128,0
255,124,0
255,120,0
255,116,0
255,116,0
255,112,0
255,109,0
255,106,0
255,104,0
255,100,0
255,96,0
255,92,0
255,89,0
255,87,0
255,84,0
255,80,0
255,76,0
255,73,0
255,72,0
255,68,0
255,65,0
255,63,0
255,60,0
255,56,0
255,52,0
255,49,0
255,46,0
255,44,0
255,40,0
255,36,0
255,32,0
255,29,0
255,27,0
255,24,0
255,22,0
255,19,0
255,16,0
255,12,0
255,8,0
255,5,0
255,3,0
255,0,0
251,0,0
247,0,0
244,0,0
241,0,0
239,0,0
235,0,0
234,0,0
231,0,0
227,0,0
223,0,0
219,0,0
217,0,0
214,0,0
211,0,0
207,0,0
203,0,0
200,0,0
198,0,0
195,0,0
191,0,0
190,0,0
187,0,0
183,0,0
179,0,0
176,0,0
173,0,0
171,0,0
167,0,0
163,0,0
159,0,0
157,0,0
154,0,0
151,0,0
147,0,0
147,0,0
143,0,0
139,0,0
135,0,0
133,0,0
130,0,0
"""


def gradient_hist(ax,hist,xbins,ybins):
    vmax=max(1,float(hist.max()));norm=colors.Normalize(0,vmax)
    polys=[];fills=[]
    for ix in range(hist.shape[0]):
        for iy in range(hist.shape[1]):
            z=float(hist[ix,iy]);x0=xbins[ix];x1=x0+(xbins[ix+1]-x0)*.85
            y0=ybins[iy];y1=y0+(ybins[iy+1]-y0)*.85
            tx=(x1-x0)*.004;ty=(y1-y0)*.004;eps=max(vmax*1e-6,1e-6)
            floor_x1 = xbins[ix+1] if ix < hist.shape[0]-1 else x1
            floor_y1 = ybins[iy+1] if iy < hist.shape[1]-1 else y1
            polys.append([(x0,y0,-eps),(floor_x1,y0,-eps),
                          (floor_x1,floor_y1,-eps),(x0,floor_y1,-eps)])
            fills.append(cmap(0))
            def add(poly,fill):polys.append(poly);fills.append(fill)
            def topframe(zz):
                black=(0,0,0,1)
                add([(x0,y0,zz),(x1,y0,zz),(x1,y0+ty,zz),(x0,y0+ty,zz)],black)
                add([(x0,y1-ty,zz),(x1,y1-ty,zz),(x1,y1,zz),(x0,y1,zz)],black)
                add([(x0,y0,zz),(x0+tx,y0,zz),(x0+tx,y1,zz),(x0,y1,zz)],black)
                add([(x1-tx,y0,zz),(x1,y0,zz),(x1,y1,zz),(x1-tx,y1,zz)],black)
            if z<=0:
                polys.append([(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0)])
                fills.append(cmap(0));
                topframe(eps)
                continue
            n=max(2,int(np.ceil(256*z/vmax)))
            zz=np.linspace(0,z,n+1)
            for a,b in zip(zz[:-1],zz[1:]):
                rgba=cmap(norm((a+b)/2))
                polys.extend([
                    [(x0,y0,a),(x1,y0,a),(x1,y0,b),(x0,y0,b)],
                    [(x1,y0,a),(x1,y1,a),(x1,y1,b),(x1,y0,b)],
                    [(x1,y1,a),(x0,y1,a),(x0,y1,b),(x1,y1,b)],
                    [(x0,y1,a),(x0,y0,a),(x0,y0,b),(x0,y1,b)]])
                fills.extend([rgba]*4)
                for xstart,xend in [(x0,x0+tx),(x1-tx,x1)]:
                    for yf in [y0-ty*.001,y1+ty*.001]:
                        add([(xstart,yf,a),(xend,yf,a),(xend,yf,b),(xstart,yf,b)],(0,0,0,1))
                for ystart,yend in [(y0,y0+ty),(y1-ty,y1)]:
                    for xf in [x0-tx*.001,x1+tx*.001]:
                        add([(xf,ystart,a),(xf,yend,a),(xf,yend,b),(xf,ystart,b)],(0,0,0,1))
            polys.append([(x0,y0,z),(x1,y0,z),(x1,y1,z),(x0,y1,z)]);fills.append(cmap(norm(z)))
            topframe(z+eps)
    pc=Poly3DCollection(polys,facecolors=fills,edgecolors='none',linewidths=0,antialiased=False,zsort='average')
    ax.add_collection3d(pc)
    ax.auto_scale_xyz([xbins[0],xbins[-2]+.85*(xbins[-1]-xbins[-2])],
                      [ybins[0],ybins[-2]+.85*(ybins[-1]-ybins[-2])],[0,vmax])
    return norm



def main():
    global cmap
    folder = Path(__file__).resolve().parent
    csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / 'SolarPrediction(2).csv'
    if not csv_file.exists():
        csv_file = folder / 'SolarPrediction.csv'
    if not csv_file.exists():
        raise FileNotFoundError('Put SolarPrediction(2).csv beside Bivariate.py or pass its path.')

    df = pd.read_csv(csv_file)
    df = df.sort_values('UNIXTime', kind='stable').reset_index(drop=True)
    local_time = pd.to_datetime(df.UNIXTime, unit='s', utc=True).dt.tz_convert('Pacific/Honolulu')
    df['time_of_day'] = local_time.dt.hour * 60 + local_time.dt.minute + local_time.dt.second / 60
    df['month'] = local_time.dt.month
    df['daylength'] = (pd.to_timedelta(df.TimeSunSet) - pd.to_timedelta(df.TimeSunRise)).dt.total_seconds() / 60
    df = df.loc[df['Radiation'] > 5]
    X = df[Features].to_numpy(dtype='float64')
    y = df['Radiation'].to_numpy(dtype='float64')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('The dataset has missing or invalid values.')

    palette = np.loadtxt(StringIO(PALETTE), delimiter=',', dtype=np.uint8) / 255.0
    cmap = colors.LinearSegmentedColormap.from_list('Original_article_jet_RGB', palette, N=1024)
    regular_path = Path('C:/Windows/Fonts/arial.ttf')
    bold_path = Path('C:/Windows/Fonts/arialbd.ttf')
    regular = (font_manager.FontProperties(fname=str(regular_path), size=7.5)
               if regular_path.is_file() else
               font_manager.FontProperties(family='DejaVu Sans', size=7.5))
    bold = (font_manager.FontProperties(fname=str(bold_path), size=9.5)
            if bold_path.is_file() else
            font_manager.FontProperties(family='DejaVu Sans', weight='bold', size=9.5))

    result = Image.new('RGB', (2880, 4380), 'white')
    for index, (feature, label) in enumerate(zip(P_Features, P_Lables)):
        values = X[:, Features.index(feature)]
        irradiance_edges = np.linspace(0, max(200, float(np.quantile(y, .995))), 16)
        low, high = np.quantile(values, [.005, .995])
        high = high if high > low else low + 1
        feature_edges = np.linspace(low, high, 16)
        hist, yy, xe = np.histogram2d(y, values, bins=[irradiance_edges, feature_edges])
        display_edges = xe / 60.0 if feature == 'time_of_day' else xe

        fig = plt.figure(figsize=(4.8, 3.65), dpi=300)
        ax = fig.add_axes([-.015, .07, .88, .91], projection='3d', proj_type='ortho')
        norm = gradient_hist(ax, hist.T, display_edges, yy)
        ax.view_init(**RefView)
        ax.set_box_aspect(Ref_BoxAspect)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(200))
        z_ticks = mticker.MaxNLocator(nbins=7, integer=True).tick_values(0, max(1, float(hist.max())))
        z_ticks = z_ticks[z_ticks >= 0]
        ax.set_zlim(0, float(z_ticks[-1]))
        ax.set_zticks(z_ticks)
        if feature == 'Pressure':
            ax.set_xticks([30.3, 30.4, 30.5])
        elif feature == 'time_of_day':
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        elif feature == 'Temperature':
            ax.set_xticks([40, 50, 60])
        elif feature == 'Humidity':
            ax.set_xticks([20, 40, 60, 80, 100])
        elif feature == 'WindDirection(Degrees)':
            ax.set_xticks([100, 200, 300])
        elif feature == 'Speed':
            ax.set_xticks([0, 5, 10, 15])
        elif feature == 'month':
            ax.set_xticks([9, 10, 11, 12])
        elif feature == 'daylength':
            ax.set_xticks([660, 680, 700, 720, 740])
        ax.tick_params(labelsize=7.5, pad=0)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_pane_color((1, 1, 1, 1))
            axis._axinfo['grid'].update({'color': (.85,.85,.85,1), 'linewidth': .4})
        for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            tick.set_fontproperties(regular)

        pos_i, pos_f, pos_z = REFERENCE_LABEL_POSITIONS[index]
        text_items = [
            fig.text(*pos_i, 'Irradiance (W/m²)', fontproperties=bold, ha='center', va='center'),
            fig.text(*pos_f, label, fontproperties=bold, ha='center', va='center'),
            fig.text(*pos_z, 'Number of occurrences', fontproperties=bold,
                     ha='center', va='center', rotation=90),
        ]
        color_axis = fig.add_axes([.78, .115, .035, .81])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
        colorbar.ax.tick_params(labelsize=7.5, width=.4, length=2, pad=2)
        colorbar.outline.set_linewidth(.4)
        for tick in colorbar.ax.get_yticklabels():
            tick.set_fontproperties(regular)

        fig.canvas.draw()
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300)
        buffer.seek(0)
        with Image.open(buffer) as picture:
            result.paste(picture.convert('RGB'), ((index % 2)*1440, (index // 2)*1095))
        plt.close(fig)
        print(f'{index + 1}/8: {label}')

    filename = folder / 'Bivariate.png'
    result.save(filename)
    print(f'Saved: {filename}')


if __name__ == '__main__':
    main()
