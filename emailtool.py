import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Email:
    def __init__(self, email_addr, subject="通知") -> None:
        self.subject = subject
        self.email_addr = email_addr

    def changeaddr(self, email_addr):
        self.email_addr = email_addr

    def send(self, message):
        # MIMEMultipart的模式，可以在邮件里夹图片。
        msg = MIMEMultipart('related')
        # content = MIMEText('<html><body><h2>打卡成功!</h2></body></html>', 'html', 'utf-8')  # 正文
        content = MIMEText(message, 'html', 'utf-8')
        # msg = MIMEText(content)
        msg.attach(content)
        msg['Subject'] = self.subject
        # 发送者邮箱
        msg['From'] = "510848570@qq.com"
        msg['To'] = self.email_addr
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 邮件服务器及端口号
        try:
            # 邮箱的校验码，需在qq邮箱网页内自行获取。
            s.login("510848570@qq.com", "rddkjicvlfeycaeh")
            s.sendmail("510848570@qq.com", self.email_addr, msg.as_string())
            print("-----邮件发送成功-----")
        except:
            print("-----邮件发送失败-----")
        finally:
            s.quit()
