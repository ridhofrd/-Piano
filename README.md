Virtual Piano dengan OpenCV and Mediapipe
Overview
    Project ini dibuat agar pengguna dapat memainkan piano digital dengan cara melakukan _finger potitioning_ pada dunia nyata dengan menggunakan media pipe untuk tracking posisi ujung jari, lalu setting ukuran piano. Piano virtual ini fleksibel karena octavnya dapat disesuaikan

### Instalasi

- Clone Repository
```
https://github.com/ridhofrd/-Piano
```
- Buat Environment Conda(Opsional)
```
conda create -n VirPiano python=3.10
```
- Aktivasi Environment
```
conda activate VirPiano
```

- Install Dependencies
```
pip install -r requirements.txt
```

#### Run Aplikasi

```
python main.py

Untuk streamlit, gunakan file: streamlit_main.py
streamlit run streamlit_main.py
```

### CATATAN PENTING
Aplikasi ini bukan aplikasi original karena mengambil referensi dari repository github ini: https://github.com/NAGAMALLYSRUJAN2329/vir_piano.git

"# -Piano"