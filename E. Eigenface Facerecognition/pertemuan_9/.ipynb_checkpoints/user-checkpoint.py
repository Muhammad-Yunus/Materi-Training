import sqlite3

class User():
    def __init__(self):
        self.conn = sqlite3.connect("attendance.db")
        self.conn.row_factory = self.dict_factory
        self.c = self.conn.cursor()

    def dict_factory(self, cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    def select_user(self, search = ''):
        self.c.execute(
            """
            SELECT * FROM User WHERE Nama LIKE ?;
            """, ('%' + search + '%',))
        return self.c.fetchall()

    def create_user(self, record):
        try :
            self.c.execute(
            """
            INSERT INTO User (
                             Nama,
                             NIM,
                             JenisKelamin,
                             Umur,
                             Alamat,
                             NamaFoto,
                             PredictionId)
                             VALUES (
                             ?, ?, ?, ?, ?, ?, ?
                             )
            """, (record['Nama'],
                 record['NIM'],
                 record['JenisKelamin'],
                 record['Umur'],
                 record['Alamat'],
                 record['NamaFoto'],
                 record['PredictionId']))
            self.conn.commit()
            return 'sucess'
        except Exception as e:
            return 'error :', e

    def update_user(self, record):
        try :
            self.c.execute(
            """
            UPDATE User SET  Nama = ?,
                             NIM = ?,
                             JenisKelamin = ?,
                             Umur = ?,
                             Alamat = ?,
                             NamaFoto = ?,
                             PredictionId = ?
            WHERE Id = ?
            """, (record['Nama'],
                 record['NIM'],
                 record['JenisKelamin'],
                 record['Umur'],
                 record['Alamat'],
                 record['NamaFoto'],
                 record['PredictionId'],
                 record['Id']))
            self.conn.commit()
            return 'sucess'
        except Exception as e:
            return 'error :', e


    def delete_user(self, Id):
        try :
            self.c.execute(
            """
            DELETE FROM User WHERE Id = ?
            """, (Id,))
            self.conn.commit()
            return 'success'
        except Exception as e:
            return 'error :', e
        
    def get_user_in_class(self, Tanggal, NamaKelas):
        self.c.execute(
        """
        SELECT
           User.Nama,
           User.NIM,
           User.JenisKelamin,
           User.NamaFoto,
           User.JenisKelamin,
           User.PredictionId,
           Kehadiran.JamMasuk
        FROM Kehadiran
        LEFT JOIN User on User.Id = Kehadiran.UserId
        WHERE Kehadiran.Date = ? AND Kehadiran.NamaKelas = ?
        """, (Tanggal, NamaKelas))

        return self.c.fetchall() 
    
    def select_user_by_prediction_id(self, PredictionId):

        self.c.execute(
        """
        SELECT * 
        FROM User 
        WHERE User.PredictionId = ?
        """, (PredictionId,))

        return self.c.fetchone() 