import db

if __name__ == '__main__':
    conn = db.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, username, role, emp_name, created_at FROM users ORDER BY id")
        rows = cur.fetchall()
        if not rows:
            print('No users found')
        else:
            for r in rows:
                try:
                    print(dict(r))
                except Exception:
                    print(r)
        cur.close()
    finally:
        conn.close()
