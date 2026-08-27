Set WshShell = CreateObject("WScript.Shell")
Set FileSystem = CreateObject("Scripting.FileSystemObject")

ProjectDir = FileSystem.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = ProjectDir

PythonExe = ProjectDir & "\.venv\Scripts\python.exe"
If Not FileSystem.FileExists(PythonExe) Then
	PythonExe = "python"
End If

PythonCommand = """" & PythonExe & """ """ & ProjectDir & "\translate.py" & """ --server"
WshShell.Run PythonCommand, 0, False

WScript.Sleep 8000

ExecutableCommand = """" & ProjectDir & "\gadz.exe" & """"
WshShell.Run ExecutableCommand, 1, False