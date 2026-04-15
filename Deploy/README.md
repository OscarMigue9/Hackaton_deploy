# Deploy local para inferencia de cacao

Esta carpeta contiene todo lo necesario para correr la interfaz local de inferencia.

## Estructura recomendada

- app.py
- requirements.txt
- README.md
- train.ipynb
- models/
  - yolo11n.pt
  - yolo11n-seg.pt

## Requisitos

- Python 3.10 o superior.

Instalacion:

```bash
pip install -r requirements.txt
```

## Ejecutar en local

Desde la carpeta Deploy:

```bash
streamlit run app.py
```

Luego abre la URL local que muestra Streamlit (normalmente http://localhost:8501).

## Flujo YOLO deteccion

1. Selecciona YOLO Deteccion (Ultralytics).
2. Elige variante de modelo (n/s/m).
3. Sube imagenes.
4. Ejecuta inferencia.

Notas:

- No se suben pesos manualmente en la interfaz.
- Si no existe el peso seleccionado, se descarga automaticamente y se guarda en models/.
- El resultado se muestra con bounding boxes.

## Flujo PyTorch personalizado por rutas locales

1. Selecciona PyTorch Personalizado (.py + .pth).
2. Ingresa ruta local al archivo .py del modelo.
3. Ingresa ruta local al archivo .pth de pesos.
4. Opcional: define nombre de clase o funcion constructora.
5. Sube imagenes y ejecuta inferencia.

Por ahora las rutas pueden quedarse vacias hasta que tengas el modelo final.

## Nota de commit

Mantener los artefactos de deploy dentro de esta carpeta reduce conflictos de rutas al hacer commit y despliegue.
