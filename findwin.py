from pywinauto import findwindows
from pywinauto import application
from pywinauto import mouse, keyboard

# 현재 윈도우 화면에 있는 프로세스 목록 리스트를 반환한다.
# 리스트의 각 요소는 element 객체로 프로세스 id, 핸들값, 이름 등의 정보를 보유한다.
procs = findwindows.find_elements()

# for proc in procs:
#     print(f"{proc} / 프로세스 : {proc.process_id}")


app = application.Application(backend="uia")

# app.start("C:/util/EbSynth-Beta-Win/EbSynth.exe")
app.connect(title_re="EbSynth Beta")
print(app)

# 컨트롤 요소 출력
dlg = app["EbSynth Beta"]  # 변수에 노트패드 윈도우 어플리케이션 객체를 할당
# dlg.print_control_identifiers()

# dlg.set_focus()
# dlg.type_keys("^o")
keyboard.send_keys("^o")
# mouse.move(coords=(150, 50))
# mouse.click()


# app.kill()
