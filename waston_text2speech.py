from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
apikey = ""
url = ""
api = IAMAuthenticator(apikey)
text_2_speech = TextToSpeechV1(authenticator = api)
text_2_speech.set_service_url(url)

with open("Alert.mp3", "wb") as audio_file:
	audio_file.write(text_2_speech.synthesize("Please wear your mask immediately!", accept = "audio/mp3").get_result().content)