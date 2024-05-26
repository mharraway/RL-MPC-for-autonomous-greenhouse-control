from SHARED.params import *
from SHARED.aux_functions import *
from SHARED.noise_model import *
import random



# Weather Data
# weather_data, total_seconds = load_weather_data("Weather Data/seljaarhires.csv")
weather_data, total_seconds = load_weather_data("Weather Data/outdoorWeatherWurGlas2014.csv")
noisy = noise_model(scale=noise_scale)

#Seed

torch.manual_seed(GlobalSeed)
np.random.seed(GlobalSeed)
random.seed(GlobalSeed)
os.environ['PYTHONHASHSEED'] = str(GlobalSeed)
torch.cuda.manual_seed(GlobalSeed)
torch.backends.cudnn.deterministic = True