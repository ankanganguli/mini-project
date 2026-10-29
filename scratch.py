from backend.app.services.text_service import text_service
print("Real News:", text_service.predict("The New York Times reports that the president has signed a new bill into law on Friday morning."))
print("Fake News:", text_service.predict("ALIENS LANDED IN WASHINGTON DC AND ATE THE PRESIDENT!!!"))
