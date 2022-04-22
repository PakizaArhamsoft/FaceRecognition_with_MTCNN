import images
import time
from PyQt5 import uic
from PyQt5 import QtWidgets
from admin_interface import CreateUser
from show_info import PersonHistory
from app import train_data
from take_attendance import MainWindow


class Start_Page(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.window = uic.loadUi("main.ui", self)
        self.action_Add_User.triggered.connect(self.show_add_user_page)
        self.action_Show_Info.triggered.connect(self.show_info_page)
        self.action_Train_Data.triggered.connect(self.training_data)
        self.action_Take_Attendance.triggered.connect(self.show_take_attend_page)


    def show_add_user_page(self):
        self.close()
        add_user = CreateUser()
        add_user.show()
    
    def show_info_page(self):
        self.close()
        ui = PersonHistory()
        ui.show()

    def training_data(self):
        train_data()
        self.msg.setText("Done!!")
        time.sleep(30)
        self.close()

    def show_take_attend_page(self):
        self.close()
        Root = MainWindow()
        Root.show()

    

if __name__ == "__main__":
    import sys 
    app = QtWidgets.QApplication(sys.argv)
    ui = Start_Page()
    ui.show()
    sys.exit(app.exec_())