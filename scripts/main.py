import yaml
from trainer import DCGAN
from pprint import pprint

def main():
	config_path = '../config/config.yml'
	with open(config_path, 'r') as stream:
		try:
			config = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)
	print('Config loaded from: {}'.format(config_path))
	pprint(config)
	DCGAN(**config)

if __name__ == '__main__':
	main()
