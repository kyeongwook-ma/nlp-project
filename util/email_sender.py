import ssl

ssl_context = ssl.create_default_context()

ssl_context.check_hostname = False

ssl_context.verify_mode = ssl.CERT_NONE

from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(subject, from_email,
               to_email, basic_text,
               password='ttfrstrmmbfuikmp'):
    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = from_email
    message['To'] = to_email

    message.attach(MIMEText(basic_text, 'plain'))
    msg_body = message.as_string()

    server = SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(message['From'], password)
    server.sendmail(message['From'], message['To'], msg_body)

    server.quit()
