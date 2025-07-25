# Documentación del Preprocesamiento de RL

Este proyecto entrena agentes de Reinforcement Learning (RL) en el entorno **SonicTheHedgehog-Sms** usando Gym Retro y wrappers inspirados en Atari.

## Archivos principales

- **main.py**: Script que crea y ensambla el entorno con todos los wrappers antes de lanzar el bucle de interacción.
- **preprocess\_frame.py**: Wrapper personalizado que convierte cada frame RGB a escala de grises, lo redimensiona y normaliza.

## Pipeline de Preprocesamiento

1. **SonicDiscretizer** (main.py)\
   Convierte los botones multibit en un espacio de acciones discretas, reduciendo la combinatoria de acciones posibles.

2. **MaxAndSkipEnv** (`skip=4`)\
   Omite 4 frames entre decisiones de acción y aplica un max-pooling de píxeles en cada par para eliminar parpadeos.

3. **PreprocessFrame** (preprocess\_frame.py)

   - Convierte a **grayscale**.
   - Redimensiona a **84×84** píxeles.
   - Normaliza valores a rango **[0,1]**.
   - Devuelve **shape** `(84,84,1)`, tipo `float32`.

4. **FrameStack** (`num_stack=4`)\
   Apila los últimos 4 frames preprocesados para capturar información de movimiento y velocidad.

5. **ClipRewardEnv**\
   Recorta la recompensa a –1, 0 o +1 para estabilizar la señal de aprendizaje.

6. **FireResetEnv**\
   Automatiza la pulsación del botón "FIRE" en cada reset cuando el juego lo requiere.

## Resultados de Debug

Ejemplo de salida por paso:

```
Step 1326 → shape=(84, 84, 4), dtype=float32, min=0.075, max=0.925, reward=0.0, info={'lives': 3, 'rings': 0, 'score': 0}, action=5
```

Nota que **shape** termina en 4, reflejando el apilamiento de frames.

## Dependencias

- Python ≥ 3.7
- gym-retro
- gym
- stable-baselines3
- opencv-python
- numpy

Instalación rápida:

```
pip install gym gym-retro stable-baselines3 opencv-python numpy
```

## Uso

```bash
python main.py
```

Cierra la ventana con la tecla **q** o cerrando la ventana de OpenCV.

---