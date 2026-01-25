import socket
import pyaudio
import threading

HOST = '0.0.0.0'
PORT = 50005
RATE = 16000
CHUNK = 1024

def start_server():
    p = pyaudio.PyAudio()

    # Stream to play NAO audio
    out_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True)
    
    # Stream to record Mac mic
    in_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Waiting for NAO...")
    conn, addr = server_socket.accept()

    def send_to_nao():
        """Reads Mac mic and sends to Linux"""
        print("Mac recording active. Sending audio to NAO...")
        try:
            while True:
                # Capture Mac mic data
                mic_data = in_stream.read(CHUNK, exception_on_overflow=False)
                conn.sendall(mic_data)
        except:
            pass

    # Start sending thread
    sender_thread = threading.Thread(target=send_to_nao)
    sender_thread.daemon = True
    sender_thread.start()

    try:
        print("Receiving audio from NAO...")
        while True:
            # Receive NAO mic data
            data = conn.recv(4096)
            if not data:
                break
            out_stream.write(data)
    except KeyboardInterrupt:
        print("Closing...")
    finally:
        conn.close()
        server_socket.close()
        out_stream.stop_stream()
        out_stream.close()
        in_stream.stop_stream()
        in_stream.close()
        p.terminate()

if __name__ == "__main__":
    start_server()