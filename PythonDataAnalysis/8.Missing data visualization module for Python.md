```
https://github.com/ResidentMario/missingno
```
Google Colab上的實測
```
!pip install quilt

!quilt install ResidentMario/missingno_data
```

```
import missingno as msno

from quilt.data.ResidentMario import missingno_data

%matplotlib inline
collisions = missingno_data.nyc_collision_factors()
collisions = collisions.replace("nan", np.nan)

msno.matrix(collisions.sample(250))
```
