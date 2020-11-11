import cv2
import ssl
import numpy as np
import pyautogui
import imutils
import smtplib

from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart  # New line
from email.mime.base import MIMEBase  # New line
from email import encoders  # New line


filename = pyautogui.screenshot()
filename = cv2.cvtColor(np.array(filename), cv2.COLOR_RGB2BGR)
cv2.imwrite("Screenshot.png", filename)
filename = 'Screenshot.png'

def email_alert():
    sender_email = ""
    sender_name = 'Mask detector ALERT'
    password = ""
    receiver_emails = ['', '']
    receiver_names = ['', '']
    email_body = '''No Mask Alert!\nThe system detected a person without mask.'''

    for receiver_email, receiver_name in zip(receiver_emails, receiver_names):
        print("Sending the email...")
        msg = MIMEMultipart()
        msg['To'] = formataddr((receiver_name, receiver_email))
        msg['From'] = formataddr((sender_name, sender_email))
        msg['Subject'] = 'Hello, ' + receiver_name        
        msg.attach(MIMEText(email_body, 'html'))

        try:
            # Open PDF file in binary mode
            with open(filename, 'rb') as attachment:
                            part = MIMEBase("application", "octet-stream")
                            part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header as key/value pair to attachment part
            part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {filename}",
            )

            msg.attach(part)
        except Exception as e:
                print(f'Oh no! We didn\'t found the attachment!\n{e}')
                break

        try:
                # Creating a SMTP session | use 587 with TLS, 465 SSL and 25
                server = smtplib.SMTP('smtp.gmail.com', 587)
                # Encrypts the email
                context = ssl.create_default_context()
                server.starttls(context=context)
                # We log in into our Google account
                server.login(sender_email, password)
                # Sending email from sender, to receiver with the email body
                server.sendmail(sender_email, receiver_email, msg.as_string())
                print('Email sent!')
        except Exception as e:
                print(f'Oh no! Something bad happened!\n{e}')
                break
        finally:
                print('Closing the server...')
                server.quit()


if __name__ == '__main__':
    email_alert()

