@portal.route("/portal/signup", methods=["POST"])
def signup():
    d = request.get_json(force=True, silent=True) or request.form
    name = (d.get("name") or "").strip()
    username = (d.get("username") or "").strip()
    password = d.get("password") or ""
    if not name or not username or len(password) < 6:
        return jsonify({"error":"Name, username and password (>=6) required"}), 400
    try:
        auth_db.create_login(username, password, role="employee", emp_name=name)
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error":"Username already taken"}), 400